from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
base_dir = Path(__file__).resolve().parent

def get_transforms(img_size: int = 64, augmented: bool = False):
    if augmented:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return train_transform, eval_transform


class FilteredImageFolder(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, classes=None, class_to_idx=None):
        self.samples = samples
        self.transform = transform
        self.classes = classes or []
        self.class_to_idx = class_to_idx or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def extract_group_id(path: str) -> str:
    # 00_image.000945_12_34_56_78.jpg
    # group id -> 00_image.000945
    filename = Path(path).stem
    parts = filename.split("_")

    if len(parts) < 2:
        return filename

    camera = parts[0]
    stem = parts[1]
    return f"{camera}_{stem}"


def grouped_train_val_split(samples, val_ratio=0.2, seed=42):
    group_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, (path, _) in enumerate(samples):
        group_id = extract_group_id(path)
        group_to_indices[group_id].append(idx)

    group_ids = list(group_to_indices.keys())

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(group_ids), generator=generator).tolist()
    group_ids = [group_ids[i] for i in perm]

    total_samples = len(samples)
    target_val_size = int(total_samples * val_ratio)

    train_indices = []
    val_indices = []
    val_count = 0

    for gid in group_ids:
        group_indices = group_to_indices[gid]
        if val_count < target_val_size:
            val_indices.extend(group_indices)
            val_count += len(group_indices)
        else:
            train_indices.extend(group_indices)

    return train_indices, val_indices


def get_dataloaders(
    data_root: str = None,
    batch_size: int = 32,
    img_size: int = 64,
    val_ratio: float = 0.2,
    augmented: bool = False,
    num_workers: int = 2,
):
    if data_root is None:
        data_root = BASE_DIR / "cropped_belgiumts_classid"
    else:
        data_root = Path(data_root)
        
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_transform, eval_transform = get_transforms(img_size=img_size, augmented=augmented)

    raw_train_dataset = datasets.ImageFolder(root=str(train_dir), allow_empty=True)
    raw_test_dataset = datasets.ImageFolder(root=str(test_dir), allow_empty=True)

    # 只保留 train/test 共同拥有的类别，避免 target_names 不一致
    train_classes = set(raw_train_dataset.classes)
    test_classes = set(raw_test_dataset.classes)
    common_classes = sorted(list(train_classes & test_classes))

    if len(common_classes) == 0:
        raise ValueError("No common classes found between train and test.")

    print(f"Found {len(common_classes)} common classes.")

    common_class_to_idx = {cls_name: i for i, cls_name in enumerate(common_classes)}

    def remap_samples(dataset, common_classes, common_class_to_idx):
        old_idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        remapped = []
        for path, old_label in dataset.samples:
            class_name = old_idx_to_class[old_label]
            if class_name in common_classes:
                remapped.append((path, common_class_to_idx[class_name]))
        return remapped

    train_samples = remap_samples(raw_train_dataset, common_classes, common_class_to_idx)
    test_samples = remap_samples(raw_test_dataset, common_classes, common_class_to_idx)

    full_train_dataset = FilteredImageFolder(
        samples=train_samples,
        transform=train_transform,
        classes=common_classes,
        class_to_idx=common_class_to_idx,
    )

    full_train_eval_dataset = FilteredImageFolder(
        samples=train_samples,
        transform=eval_transform,
        classes=common_classes,
        class_to_idx=common_class_to_idx,
    )

    test_dataset = FilteredImageFolder(
        samples=test_samples,
        transform=eval_transform,
        classes=common_classes,
        class_to_idx=common_class_to_idx,
    )

    train_indices, val_indices = grouped_train_val_split(
        train_samples,
        val_ratio=val_ratio,
        seed=42
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_eval_dataset, val_indices)

    print(f"Total train samples: {len(train_samples)}")
    print(f"Grouped train samples: {len(train_indices)}")
    print(f"Grouped val samples: {len(val_indices)}")
    print(f"Total test samples: {len(test_samples)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, common_classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)
    print("Num classes:", len(classes))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))