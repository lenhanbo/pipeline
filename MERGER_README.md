# Handwriting Merger baseline

`MergerNet` is an isolated ViT image-restoration baseline. Its continuation pretrain
keeps the complete crop visible and learns to undo mild rotation (at most three
degrees by default), translation, shear, scale, blur, and additive noise. It then
learns to repair a 32-pixel seam after component-level corruptions. Random
occlusion is disabled by default; the requested center gap is still removed. It
does not yet generate characters.

The local merger is a pure residual ViT: it patchifies the `32x64` two-character
window into `4x4` tokens, prepends the final causal prefix token, and projects
the output tokens directly back to pixels. There is no U-Net or skip decoder.
Training intentionally uses only two masked L1 terms: reconstruction against the
clean target over both patches, and identity against the input over the stable
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

Every seam sample also removes a configurable two-pixel vertical band centered
on the split (one column per side). This guarantees that even an otherwise clean
pair requires a real boundary reconstruction.

## Data contract

The configured data root must contain `train_gt.txt`, `val_gt.txt`, and the
`words/` directory. Ground-truth rows use `relative/image.png<TAB>text`, matching
the IAM data already present in this workspace.

The model consumes a grayscale image in `[-1,1]`, a restoration/corruption mask,
and an edit mask. Continuation uses the whole valid crop as the editable region;
it does not paint over or delete the input. Only pixels selected by the edit mask
can change, so hard blending makes the outside-mask error exactly zero.

## Commands

From `pipeline/`:

```powershell
python -m unittest tests.test_merger
python train_merger.py --config configs/merger_iam.yml --stage continuation
python train_merger.py --config configs/merger_iam.yml --stage seam `
  --resume runs/merger_32x32/continuation/best_ssim.pth
python evaluate_merger.py --config configs/merger_iam.yml --stage seam `
  --checkpoint runs/merger_32x32/seam/best_seam.pth
```

Use `--data-root /kaggle/input/.../iam_dataset` and
`--output-dir /kaggle/working/merger_32x32` on Kaggle. Checkpoints are self-contained
and include the resolved configuration, optimizer, scheduler, scaler, and step.
Checkpoints from the older `32x16` character-patch layout are shape-incompatible;
start continuation training from scratch after switching to this configuration.

Evaluation writes fixed preview grids and JSON metrics for clean seams, masks,
translation, rotation, shear/scale, photometric corruption, and their combination.
The report also includes percentage improvement over both no-op and edge-copy
baselines; positive percentages mean the learned merger is better.
