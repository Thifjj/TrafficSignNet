from pathlib import Path
import csv
import random
import shutil
import urllib.request
import zipfile

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "gtsrb"

DOWNLOAD_DIR = DATA_DIR / "downloads"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "split"

VAL_RATIO = 0.20
SEED = 42

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

FILES = [
    "GTSRB_Final_Training_Images.zip",
    "GTSRB_Final_Test_Images.zip",
    "GTSRB_Final_Test_GT.zip",
]


def download_and_extract():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        zip_path = DOWNLOAD_DIR / filename

        if not zip_path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(BASE_URL + filename, zip_path)

        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)


def save_as_png(src, dst):
    with Image.open(src) as image:
        image = image.convert("RGB")
        image.save(dst, format="PNG")


def split_train_val():
    source = RAW_DIR / "GTSRB" / "Final_Training" / "Images"

    train_dir = SPLIT_DIR / "train"
    val_dir = SPLIT_DIR / "val"

    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)

    random.seed(SEED)

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue

        images = sorted(class_dir.glob("*.ppm"))

        tracks = {}

        for image in images:
            track = image.stem.split("_")[0]
            tracks.setdefault(track, []).append(image)

        track_names = list(tracks.keys())
        random.shuffle(track_names)

        n_val = max(1, int(len(track_names) * VAL_RATIO))
        val_tracks = set(track_names[:n_val])

        for track, track_images in tracks.items():
            destination = val_dir if track in val_tracks else train_dir
            destination = destination / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)

            for image in track_images:
                output = destination / f"{image.stem}.png"
                save_as_png(image, output)


def prepare_test():
    images_dir = RAW_DIR / "GTSRB" / "Final_Test" / "Images"
    csv_file = next(RAW_DIR.rglob("GT-final_test.csv"))

    test_dir = SPLIT_DIR / "test"
    shutil.rmtree(test_dir, ignore_errors=True)

    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            image = images_dir / row["Filename"]
            class_dir = test_dir / f"{int(row['ClassId']):05d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            output = class_dir / f"{image.stem}.png"
            save_as_png(image, output)


def count_images(folder):
    return len(list(folder.rglob("*.png")))


def main():
    print("=== GTSRB DATASET ===")

    download_and_extract()

    print("\nCreating train/validation split...")
    split_train_val()

    print("Preparing official test set...")
    prepare_test()

    print("\nDone.")
    print(f"Train: {count_images(SPLIT_DIR / 'train')}")
    print(f"Val  : {count_images(SPLIT_DIR / 'val')}")
    print(f"Test : {count_images(SPLIT_DIR / 'test')}")
    print(f"\nDataset saved in: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
