# Dataset

The project uses GTSRB with class directories `00000` through `00042`. Labels are taken from the directory name. Images are converted to RGB PNG during preparation.

## Layout

- `data/gtsrb/downloads/`: source ZIP archives.
- `data/gtsrb/raw/`: extracted GTSRB files.
- `data/gtsrb/processed/train/`: training split.
- `data/gtsrb/processed/val/`: track-aware validation split and INT8 calibration data.
- `data/gtsrb/processed/test/`: official test images arranged by class.
- `deploy/zcu104/dataset/`: small board bundle, 10 images per class.

## Recreate the dataset

```bash
python3 training/prepare_dataset.py
```

The script downloads the three GTSRB archives, extracts them, creates a deterministic 80/20 track-aware train/validation split with seed 42, and prepares the official test set. It may replace the existing processed train, validation, and test directories.
