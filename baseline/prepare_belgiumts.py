import csv
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image

# ========= 配置区 =========
project_root  = Path(__file__).resolve().parent.parent
data_root = project_root / "data"   
annotation_dir = data_root  / "BelgiumTSD_annotations"  
output_root = Path(__file__).resolve().parent / "cropped_belgiumts_classid"  

# 训练和测试标注文件
train_file = annotation_dir / "BTSD_training_GTclear.txt"
test_file = annotation_dir / "BTSD_testing_GTclear.txt"

# None = 自动使用所有 camera
allowed_cameras = None

min_width = 20
min_height = 20


# 设为 0 表示不过滤
min_samples_per_class = 10
# =========================


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def reset_output_dir(output_root: Path):
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def parse_annotation_file(txt_path: Path):
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=";")
        for line_num, parts in enumerate(reader, start=1):
            parts = [p.strip() for p in parts if p.strip() != ""]

            if len(parts) < 7:
                print(f"[WARN] Skip malformed line {line_num} in {txt_path.name}: {parts}")
                continue

            img_rel = parts[0]
            try:
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                class_id = int(parts[5])
                superclass_id = int(parts[6])
            except ValueError:
                print(f"[WARN] Skip invalid numeric line {line_num} in {txt_path.name}: {parts}")
                continue

            camera = img_rel.split("/")[0]
            image_name = img_rel.split("/")[-1]

            rows.append({
                "camera": camera,
                "img_rel": img_rel,
                "image_name": image_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id,
                "superclass_id": superclass_id,
            })
    return rows


def get_label_name(class_id: int, superclass_id: int = None):
    # 过滤 undefined 类
    if class_id == -1:
        return None
    return f"class_{class_id}"


def camera_allowed(camera: str) -> bool:
    if allowed_cameras is None:
        return True
    return camera in allowed_cameras


def is_valid_bbox(row) -> bool:
    x1 = int(round(row["x1"]))
    y1 = int(round(row["y1"]))
    x2 = int(round(row["x2"]))
    y2 = int(round(row["y2"]))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < min_width or crop_h < min_height:
        return False
    if x2 <= x1 or y2 <= y1:
        return False

    return True


def crop_one_image(image_path: Path, x1: float, y1: float, x2: float, y2: float):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < min_width or crop_h < min_height:
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))


def collect_class_counts(rows):
    counter = Counter()

    for row in rows:
        if not camera_allowed(row["camera"]):
            continue

        label_name = get_label_name(row["class_id"], row["superclass_id"])
        if label_name is None:
            continue

        if not is_valid_bbox(row):
            continue

        counter[label_name] += 1

    return counter


def build_allowed_class_set(train_rows, test_rows):
    if min_samples_per_class <= 0:
        return None

    train_counter = collect_class_counts(train_rows)
    test_counter = collect_class_counts(test_rows)

    allowed = set()
    all_classes = set(train_counter.keys()) | set(test_counter.keys())

    for cls in all_classes:
        if (
            train_counter.get(cls, 0) >= min_samples_per_class
            and test_counter.get(cls, 0) >= min_samples_per_class
        ):
            allowed.add(cls)

    return allowed


def process_split(rows, split_name: str, allowed_classes=None):
    split_output = output_root / split_name
    safe_mkdir(split_output)

    class_counter = Counter()
    skipped_camera = 0
    skipped_label = 0
    skipped_bbox = 0
    skipped_missing = 0
    skipped_rare = 0
    saved_count = 0

    for idx, row in enumerate(rows, start=1):
        camera = row["camera"]

        if not camera_allowed(camera):
            skipped_camera += 1
            continue

        label_name = get_label_name(row["class_id"], row["superclass_id"])
        if label_name is None:
            skipped_label += 1
            continue

        if allowed_classes is not None and label_name not in allowed_classes:
            skipped_rare += 1
            continue

        image_path = data_root / row["img_rel"]
        image_path = image_path.with_suffix('.jp2')
        if not image_path.exists():
            skipped_missing += 1
            continue

        crop = crop_one_image(
            image_path=image_path,
            x1=row["x1"],
            y1=row["y1"],
            x2=row["x2"],
            y2=row["y2"],
        )

        if crop is None:
            skipped_bbox += 1
            continue

        class_dir = split_output / label_name
        safe_mkdir(class_dir)

        stem = Path(row["image_name"]).stem
        save_name = (
            f"{camera}_{stem}_"
            f"{int(round(row['x1']))}_{int(round(row['y1']))}_"
            f"{int(round(row['x2']))}_{int(round(row['y2']))}.jpg"
        )
        save_path = class_dir / save_name

        crop.save(save_path, quality=95)
        class_counter[label_name] += 1
        saved_count += 1

        if idx % 1000 == 0:
            print(f"[{split_name}] processed {idx}/{len(rows)}")

    print(f"\n===== {split_name.upper()} SUMMARY =====")
    print(f"Saved crops: {saved_count}")
    print(f"Skipped camera: {skipped_camera}")
    print(f"Skipped label: {skipped_label}")
    print(f"Skipped bbox: {skipped_bbox}")
    print(f"Missing images: {skipped_missing}")
    print(f"Skipped rare: {skipped_rare}")
    print("Class counts:")
    for cls, cnt in sorted(class_counter.items(), key=lambda x: x[0]):
        print(f"  {cls:<18} {cnt}")
    print()

    return class_counter


def main():
    if not train_file.exists():
        raise FileNotFoundError(f"Training annotation not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Testing annotation not found: {test_file}")

    print("Resetting output directory...")
    reset_output_dir(output_root)

    print("Reading annotations...")
    train_rows = parse_annotation_file(train_file)
    test_rows = parse_annotation_file(test_file)

    print(f"Train annotations loaded: {len(train_rows)}")
    print(f"Test annotations loaded: {len(test_rows)}\n")

    if allowed_cameras is None:
        train_cams = {r['camera'] for r in train_rows}
        test_cams = {r['camera'] for r in test_rows}
        print(f"Detected train cameras: {sorted(train_cams)}")
        print(f"Detected test cameras: {sorted(test_cams)}\n")

    allowed_classes = build_allowed_class_set(train_rows, test_rows)
    if allowed_classes is not None:
        print(
            f"Keeping {len(allowed_classes)} class-id categories "
            f"with >= {min_samples_per_class} samples in both train and test.\n"
        )

    train_counter = process_split(train_rows, "train", allowed_classes=allowed_classes)
    test_counter = process_split(test_rows, "test", allowed_classes=allowed_classes)

    print("Done.")
    print(f"Output saved to: {output_root.resolve()}")

    stats_path = output_root / "class_distribution.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("TRAIN\n")
        for cls, cnt in sorted(train_counter.items(), key=lambda x: x[0]):
            f.write(f"{cls}: {cnt}\n")

        f.write("\nTEST\n")
        for cls, cnt in sorted(test_counter.items(), key=lambda x: x[0]):
            f.write(f"{cls}: {cnt}\n")

    print(f"Class distribution saved to: {stats_path.resolve()}")


if __name__ == "__main__":
    main()