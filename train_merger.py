from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from merger.losses import MergerLoss
from merger.data import merger_collate
from merger.runtime import (
    append_metrics,
    build_dataset,
    build_model,
    cosine_schedule,
    evaluate_loader,
    load_config,
    move_batch,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the end-to-end handwriting merger")
    parser.add_argument("--config", default="configs/merger_iam.yml")
    parser.add_argument("--stage", choices=("merge",), default="merge")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    seed_everything(int(config.get("seed", 123456)))

    requested_device = args.device or config.get("device", "cuda")
    device = torch.device(requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu")
    training = config["training"]
    stage_config = training[args.stage]
    max_steps = args.max_steps or int(stage_config["max_steps"])
    output_dir = Path(training["output_dir"]) / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    train_dataset = build_dataset(config, args.stage, "train", deterministic=False)
    val_dataset = build_dataset(config, args.stage, "val", deterministic=True)
    loader_options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, collate_fn=merger_collate, **loader_options
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, collate_fn=merger_collate, **loader_options
    )

    model = build_model(config).to(device)
    loss_fn = MergerLoss(**config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = int(training.get("warmup_steps", 500))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_schedule(step, warmup_steps, max_steps)
    )
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = int(training.get("gradient_accumulation", 1))
    start_step = 0
    best_ssim = float("-inf")
    best_seam = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("stage") == args.stage:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_step = int(checkpoint["step"])
            best_ssim = float(checkpoint.get("best_ssim", best_ssim))
            best_seam = float(checkpoint.get("best_seam", best_seam))
        print(f"Loaded {args.resume} at step {start_step}")

    writer = SummaryWriter(output_dir / "tensorboard")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iterator = iter(train_loader)
    progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps)
    eval_every = int(training.get("eval_every", 500))
    log_every = int(training.get("log_every", 20))
    running: dict[str, float] = {}

    for step in progress:
        step_components: dict[str, float] = {}
        for _ in range(accumulation):
            try:
                raw_batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                raw_batch = next(data_iterator)
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = model(
                    batch["left_image"],
                    batch["right_image"],
                    batch["left_corruption_mask"],
                    batch["right_corruption_mask"],
                    batch["prefix_context"],
                    batch["prefix_mask"],
                )
                naive_concat = torch.cat((batch["left_image"], batch["right_image"]), dim=-1)
                loss, components = loss_fn(
                    prediction,
                    batch["target"],
                    batch["edit_mask"],
                    batch["valid_mask"],
                    input_image=naive_concat,
                    left_preservation_mask=batch["left_preservation_mask"],
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            for name, value in components.items():
                step_components[name] = step_components.get(name, 0.0) + float(value.detach()) / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        for name, value in step_components.items():
            running[name] = running.get(name, 0.0) + value
        if (step + 1) % log_every == 0:
            means = {name: value / log_every for name, value in running.items()}
            for name, value in means.items():
                writer.add_scalar(f"train/{name}", value, step + 1)
            progress.set_postfix(loss=f"{means['loss']:.4f}")
            running.clear()

        should_evaluate = (step + 1) % eval_every == 0 or step + 1 == max_steps
        if should_evaluate:
            result = evaluate_loader(
                model,
                val_loader,
                device,
                max_batches=int(training.get("validation_batches", 50)),
                preview_path=output_dir / "previews" / f"step_{step + 1:07d}.png",
            )
            row = {"step": step + 1}
            for group, values in result.items():
                for name, value in values.items():
                    row[f"{group}_{name}"] = value
                    writer.add_scalar(f"val/{group}_{name}", value, step + 1)
            append_metrics(output_dir / "metrics.csv", row)
            save_json(output_dir / "metrics.json", row)

            checkpoint = {
                "stage": args.stage,
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_ssim": best_ssim,
                "best_seam": best_seam,
                "config": config,
            }
            torch.save(checkpoint, output_dir / "latest.pth")
            if result["model"]["ssim"] > best_ssim:
                best_ssim = result["model"]["ssim"]
                checkpoint["best_ssim"] = best_ssim
                torch.save(checkpoint, output_dir / "best_ssim.pth")
            if result["model"]["seam_discontinuity"] < best_seam:
                best_seam = result["model"]["seam_discontinuity"]
                checkpoint["best_seam"] = best_seam
                torch.save(checkpoint, output_dir / "best_seam.pth")
            model.train()

    writer.close()
    print(f"Training complete. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
