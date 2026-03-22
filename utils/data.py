import os.path as osp
import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):
    return x.mul(2).sub_(1)


def build_imagenet_dataset(data_path: str, final_reso: int = 256, hflip: bool = True, mid_reso: float = 1.125):
    mid_reso_px = round(mid_reso * final_reso)

    train_transforms = []
    if hflip:
        train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms += [
        transforms.Resize(mid_reso_px, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop(final_reso),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]

    val_transforms = [
        transforms.Resize(mid_reso_px, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop(final_reso),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]

    train_set = DatasetFolder(
        root=osp.join(data_path, 'train'),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=transforms.Compose(train_transforms),
    )
    val_set = DatasetFolder(
        root=osp.join(data_path, 'val'),
        loader=pil_loader,
        extensions=IMG_EXTENSIONS,
        transform=transforms.Compose(val_transforms),
    )
    return train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PImage.open(f).convert('RGB')
    return img
