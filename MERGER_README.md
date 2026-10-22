# Handwriting Merger baseline

`MergerNet` is a single-stage, end-to-end ViT merger. Every training batch supplies
two independent character images and the model directly generates their complete
merged canvas. Geometry, photometric noise, overlap removal, and the two-pixel
center gap are data augmentations inside the same `merge` stage; there is no
continuation pretrain or separate seam fine-tune. It does not yet generate the
individual character images upstream of the merger.

The merger is a direct ViT image generator. It receives the left `32x32` image
and right `32x32` image as two independent tensors; it never receives a spatially
pre-concatenated image. A shared patch embedder tokenizes each branch separately,
learned left/right role embeddings preserve their identity, and the final causal
prefix token conditions their joint Transformer. The decoder generates every
pixel of two new `32x32` outputs, which the model joins into the final `32x64`
canvas. There is no input-to-output residual addition, hard pixel copy, U-Net,
or skip decoder.
Training intentionally uses only two masked L1 terms: reconstruction against the
clean target over both generated patches, and identity against the naive input
concatenation over the stable
part of the left patch. SSIM, Sobel, Charbonnier, boundary, perceptual, GAN, and
CTC losses are not part of this first baseline.

This baseline now represents one generated character as a square `32x32` patch.
The merger's local image is therefore `32x64`: the last 32 pixels of the
existing prefix and the new 32-pixel character patch. A boundary point is selected
along the horizontal length of the original image; the full image height is always
kept. All pixels before that point form the complete prefix. Variable-width prefixes
are padded only for batching and split on the same 32-pixel character grid. Each character patch is encoded into
exactly one token, then the ordered token sequence is passed through a causal
Transformer. The final valid Transformer token is the prefix state used to
condition the local merger, matching the last-hidden-state idea of a language
model. That token is prepended to the local window tokens for joint Transformer
fusion; it is not merely averaged or added as a global bias. There is no prefix
pooling, and padding cannot become the final token.

The decoder predicts both adjacent patches, not only the seam strip. A dedicated
left-identity loss discourages changes to the established prefix layout, while
excluding corrupted/overlap pixels near the join. The center band is deliberately
erased to remove the direct overlap cue and force reconstruction of a coherent
connection between the two patches.

Every merge sample also removes a configurable two-pixel vertical band centered
on the split (one column per side). This guarantees that even an otherwise clean
pair requires a real boundary reconstruction.

## Data contract

The configured data root must contain `train_gt.txt`, `val_gt.txt`, and the
`words/` directory. Ground-truth rows use `relative/image.png<TAB>text`, matching
the IAM data already present in this workspace. Transcripts containing fewer
than two non-whitespace characters are excluded from training, validation, and
checkpoint analysis because they do not define a two-character merge boundary.

The model consumes two grayscale images in `[-1,1]`, one corruption mask for
each image, the complete prefix, and its padding mask. The combined `edit_mask`
is used only to select reconstruction-loss and evaluation pixels; it is not fed
to the model and does not hard-copy any input pixels into the output.

## Commands

From `pipeline/`:

```powershell
python -m unittest tests.test_merger
python train_merger.py --config configs/merger_iam.yml --stage merge
python evaluate_merger.py --config configs/merger_iam.yml --stage merge `
  --checkpoint runs/merger_e2e/merge/best_seam.pth
python analyze_merger.py `
  --checkpoint runs/merger_e2e/merge/best_seam.pth `
  --stage merge --suite combined --selection worst_noise
```

Use `--data-root /kaggle/input/.../iam_dataset` and
`--output-dir /kaggle/working/merger_e2e` on Kaggle. Checkpoints are self-contained
and include the resolved configuration, optimizer, scheduler, scaler, and step.
Checkpoints from the older staged or residual architectures are shape-incompatible
with the two-input direct generator and must not be resumed.

Evaluation writes fixed preview grids and JSON metrics for clean seams, masks,
translation, rotation, shear/scale, photometric corruption, and their combination.
The report also includes percentage improvement over both no-op and edge-copy
baselines; positive percentages mean the learned merger is better.

`analyze_merger.py` samples a configurable candidate pool from a checkpoint and
writes labeled preview pages, per-sample metrics in `samples.csv`, and a summary.
Use `--selection random` with different `--seed` values to inspect different
examples, or rank failures with `worst_mae`, `worst_seam`, and `worst_noise`.
The noise ranking measures residual error and false ink only on pixels whose clean
target is confidently background.
