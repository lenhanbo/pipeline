# Handwriting Merger baseline

`MergerNet` is an isolated masked image-restoration baseline. It first learns to
restore the right half of an IAM word crop, then learns to repair a 32-pixel seam
after rotation, translation, shear, scale, stroke, photometric, and occlusion
corruptions. It does not yet generate characters.

Every seam sample also removes a configurable two-pixel vertical band centered
on the split (one column per side). This guarantees that even an otherwise clean
pair requires a real boundary reconstruction.

## Data contract

The configured data root must contain `train_gt.txt`, `val_gt.txt`, and the
`words/` directory. Ground-truth rows use `relative/image.png<TAB>text`, matching
the IAM data already present in this workspace.

The model consumes a grayscale image in `[-1,1]`, a corruption mask, and an edit
mask. Only pixels selected by the edit mask can change; hard blending makes the
outside-mask error exactly zero.

## Commands

From `pipeline/`:

```powershell
python -m unittest tests.test_merger
python train_merger.py --config configs/merger_iam.yml --stage continuation
python train_merger.py --config configs/merger_iam.yml --stage seam `
  --resume runs/merger_iam/continuation/best_ssim.pth
python evaluate_merger.py --config configs/merger_iam.yml --stage seam `
  --checkpoint runs/merger_iam/seam/best_seam.pth
```

Use `--data-root /kaggle/input/.../iam_dataset` and
`--output-dir /kaggle/working/merger_iam` on Kaggle. Checkpoints are self-contained
and include the resolved configuration, optimizer, scheduler, scaler, and step.

Evaluation writes fixed preview grids and JSON metrics for clean seams, masks,
translation, rotation, shear/scale, photometric corruption, and their combination.
The report also includes percentage improvement over both no-op and edge-copy
baselines; positive percentages mean the learned merger is better.
