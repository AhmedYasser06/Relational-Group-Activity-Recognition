"""
Shared eval helpers for temporal models.
Each model file calls these instead of duplicating the same lines.
"""

from pathlib import Path

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from utils import model_eval, model_eval_TTA, GroupActivityDataset, group_activity_labels


BASE_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

TTA_TRANSFORMS = [
    BASE_TRANSFORM,
    A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise(),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
        ], p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]),
    A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise(),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
        ], p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]),
]


def _load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)


def _make_loader(root, config, collate_fn, batch_size=14):
    dataset = GroupActivityDataset(
        videos_path=f"{root}/{config.data['videos_path']}",
        annot_path=f"{root}/{config.data['annot_path']}",
        split=config.data['video_splits']['test'],
        labels=group_activity_labels,
        transform=BASE_TRANSFORM,
        seq=True,
        sort=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def run_eval(model, root, config, checkpoint_path, collate_fn, prefix, batch_size=14):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_model(model, checkpoint_path, device)
    loader = _make_loader(root, config, collate_fn, batch_size)

    return model_eval(
        model=model,
        data_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        path=str(Path(checkpoint_path).parent),
        prefix=prefix,
        class_names=config.model['num_clases_label']['group_activity'],
    )


def run_eval_TTA(model, root, config, checkpoint_path, collate_fn, prefix, batch_size=14):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_model(model, checkpoint_path, device)

    dataset_params = {
        'videos_path': f"{root}/{config.data['videos_path']}",
        'annot_path': f"{root}/{config.data['annot_path']}",
        'split': config.data['video_splits']['test'],
        'labels': group_activity_labels,
        'seq': True,
        'sort': True,
        'batch_size': batch_size,
        'num_workers': 4,
        'collate_fn': collate_fn,
        'pin_memory': True,
    }

    return model_eval_TTA(
        model=model,
        dataset=GroupActivityDataset,
        dataset_params=dataset_params,
        tta_transforms=TTA_TRANSFORMS,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        path=str(Path(checkpoint_path).parent),
        prefix=prefix,
        class_names=config.model['num_clases_label']['group_activity'],
    )
