# Handwriting Merger baseline

`MergerNet` is a single-stage, end-to-end ViT merger. Every training batch supplies
two independent character images and the model directly generates their complete
merged canvas. Geometry, mild ink-only random noise, and optional boundary-column
removal are data augmentations inside the same `merge` stage; there is no
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
Training currently uses only two masked MSE terms: reconstruction against the
clean target over both generated patches, and identity against the naive input
concatenation over the stable
part of the left patch. SSIM, Sobel, Charbonnier, boundary, perceptual, GAN, and
CTC losses are not part of this baseline experiment.

This baseline represents one generated character as a square `32x32` patch.
For each sample, the loader selects a horizontal interval of exactly 64 pixels
from a truncated normal distribution centered on the resized word. It retries
candidate windows until both 32-pixel halves contain visible ink, while always
retaining the full 32-pixel image height. If a short or sparse word cannot provide
two informative halves, the dataset retries another word from the same split
rather than padding one model input with an empty patch.
It splits that interval at the center into two independent `32x32` model inputs.
All pixels in the original word before the center cut form the complete prefix. Variable-width prefixes
are padded only for batching and split on the same 32-pixel character grid. Each character patch is encoded into
exactly one token, then the ordered token sequence is passed through a causal
Transformer. The final valid Transformer token is the prefix state used to
condition the local merger, matching the last-hidden-state idea of a language
model. That token is prepended to the local window tokens for joint Transformer
fusion; it is not merely averaged or added as a global bias. There is no prefix
pooling, and padding cannot become the final token.

The decoder predicts both adjacent patches, not only a seam strip. A dedicated
left-identity loss discourages changes to the established prefix layout. The left
input remains clean. On the right input, the loader may whiten a configurable
full-height strip covering the first few columns next to the cut. With the default
width of two, columns 0 and 1 of the right patch are completely removed before
centering and affine augmentation. The `seam_mask` is only a two-column evaluation
region around the join and does not modify either model input.

Rotation, translation, shear, and scaling are applied to the right patch on a
large padded canvas. Before affine, the right-side ink bounding box is treated
like an independently generated character: it is fit into a margin and centered
inside `32x32`. Rotation, scale, and translation therefore produce a real visible
layout change instead of being canceled when the original crop touched a border.
The transformed ink bounding box is then kept inside `32x32`, so augmentation
cannot clip strokes at the patch border. Photometric
augmentation applies only mild Gaussian noise to ink opacity rather than raw
brightness. It applies no blur, contrast jitter, or morphology. Consequently,
paper remains exactly white and cannot acquire a different gray background from
the left patch.

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
python preview_merger_data.py --suite combined --num-samples 16
python visualize_merger_steps.py --suite combined --indices 55464 666 44704
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

`preview_merger_data.py` needs no checkpoint. It writes labeled input pages and
`samples.csv` under `runs/merger_data_preview/`, making it the quickest way to
inspect the random 64-pixel window, independent left/right inputs, boundary erase,
geometry, and background consistency before starting a Kaggle run. Use another
`--seed` to draw a different reproducible set.

`visualize_merger_steps.py` traces individual samples through all data
preparation views: normalized source word and selected 64px window, 32/32 split,
boundary erasure, ink-only random noise, standalone centering, affine
geometry, final two-input diagnostic view, clean target, and their difference.
It shows rotation-only and pre-translation diagnostic panels separately so a
small random angle cannot be confused with translation. Training still applies
the final combined affine in one interpolation operation.
Pass exact dataset positions with `--indices`, or omit it and use `--num-samples`
plus `--seed` for reproducible random examples. Trace tensors are disabled in
normal training and therefore add no training memory overhead.
