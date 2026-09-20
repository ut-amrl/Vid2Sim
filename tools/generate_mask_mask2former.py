"""Dynamic-object masks for a Vid2Sim ``seq_path``, via COCO-panoptic Mask2Former.

Stands in for upstream's ``generate_mask.sh``. Vid2Sim offers two maskers, DEVA and
Grounded-SAM-2, both driven by the text prompt
``person.pedestrian.cyclist.child.adult.bag.backpack....`` -- i.e. the target is "people
and what they carry". A panoptic segmenter reaches the same target directly, with
closed-vocabulary labels instead of an open-vocabulary prompt, and without the two sets
of detector checkpoints that Grounded-SAM-2 needs.

Writes the same mask twice because the two consumers disagree on naming:

* ``masks/00001.jpg``      -- Vid2Sim's ``dataset_readers.py`` derives the mask path by
  string-replacing ``images`` with ``masks``, so the *image* extension is kept. The
  bytes are PNG regardless; PIL dispatches on content, not on the name.
* ``colmap_masks/00001.jpg.png`` -- COLMAP's ``--ImageReader.mask_path`` wants the full
  image filename plus ``.png``.

Convention for both: 255 = keep, 0 = ignore.

    micromamba run -n vid2sim-recon python tools/generate_mask_mask2former.py \
        --seq data/<clip> --labels person
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

MODEL_NAME = 'facebook/mask2former-swin-large-coco-panoptic'


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--seq', required=True, help='seq_path containing images/')
  ap.add_argument('--labels', nargs='+',
                  default=['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'],
                  help='COCO-panoptic classes treated as dynamic.')
  ap.add_argument('--dilate', type=int, default=9,
                  help='Dilation of the ignore region (px). Segmentation boundaries sit '
                       'a pixel or two inside the silhouette, and the leftover rim is '
                       'exactly the high-gradient edge SfM likes to latch onto.')
  ap.add_argument('--batch-size', type=int, default=8)
  args = ap.parse_args()

  seq = pathlib.Path(args.seq)
  imgs = sorted((seq / 'images').glob('*.jpg'))
  if not imgs:
    raise SystemExit(f'no images in {seq / "images"}')
  mask_dir, colmap_dir = seq / 'masks', seq / 'colmap_masks'
  mask_dir.mkdir(parents=True, exist_ok=True)
  colmap_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)
  model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME).to(device).eval()

  label2id = {v.lower(): k for k, v in model.config.id2label.items()}
  dyn = {label2id[n] for n in args.labels if n in label2id}
  print(f'dynamic classes {sorted(args.labels)} -> ids {sorted(dyn)}')

  kernel = np.ones((args.dilate, args.dilate), np.uint8) if args.dilate > 0 else None
  covered = []

  for i in tqdm.tqdm(range(0, len(imgs), args.batch_size), desc='masking'):
    batch = imgs[i:i + args.batch_size]
    pil = [Image.open(p).convert('RGB') for p in batch]
    inputs = processor(images=pil, return_tensors='pt').to(device)
    with torch.no_grad():
      out = model(**inputs)
    panoptic = processor.post_process_panoptic_segmentation(
      out, target_sizes=[p.size[::-1] for p in pil])

    for path, res in zip(batch, panoptic):
      seg = res['segmentation'].cpu().numpy()
      ignore = np.zeros(seg.shape, bool)
      for info in res['segments_info']:
        if info['label_id'] in dyn:
          ignore |= seg == info['id']
      if kernel is not None and ignore.any():
        ignore = cv2.dilate(ignore.astype(np.uint8), kernel, 1).astype(bool)

      # The stereo extractor pre-writes a validity mask marking the wedges rectification
      # leaves blank. Those are just as unusable as a pedestrian, so keep both.
      prior = cv2.imread(str(mask_dir / path.name), cv2.IMREAD_GRAYSCALE)
      if prior is not None:
        ignore |= prior == 0

      keep = np.where(ignore, 0, 255).astype(np.uint8)
      covered.append(float(ignore.mean()))
      cv2.imwrite(str(mask_dir / path.name), keep)          # PNG bytes, .jpg name
      cv2.imwrite(str(colmap_dir / f'{path.name}.png'), keep)

  covered = np.array(covered)
  print(f'{len(imgs)} frames | ignored pixels: mean {covered.mean():.2%} '
        f'max {covered.max():.2%} | frames with any dynamic content: '
        f'{(covered > 0).sum()}/{len(covered)}')


if __name__ == '__main__':
  main()
