import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet50_backbone():
    """Returns ResNet50 up to the last pooling layer (output: 2048-d per image)."""
    return nn.Sequential(
        *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
    )


class PersonClassifier(nn.Module):
    """
    Stage-1 model: fine-tuned ResNet50  2048-d classifier.
    Input shape: (batch, num_players, C, H, W)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.resnet50 = build_resnet50_backbone()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        b, bb, c, h, w = x.shape
        x = x.view(b * bb, c, h, w)
        x = self.resnet50(x).view(b * bb, -1)
        return self.fc(x)


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def collate_fn(batch, seq=False):
    """
    Pads player dimension to 12 per sample.
    Set seq=True if clips have shape (players, frames, C, H, W).
    """
    clips, labels = zip(*batch)
    max_players = 12
    padded = []

    for clip in clips:
        n = clip.size(0)
        if n < max_players:
            if seq:
                pad = torch.zeros(max_players - n, clip.size(1), clip.size(2), clip.size(3), clip.size(4))
            else:
                pad = torch.zeros(max_players - n, clip.size(1), clip.size(2), clip.size(3))
            clip = torch.cat([clip, pad], dim=0)
        padded.append(clip)

    return torch.stack(padded), torch.stack(labels)


def collate_fn_seq(batch):
    """collate_fn for sequential clips uses the last frame's label."""
    clips, labels = zip(*batch)
    max_players = 12
    padded = []

    for clip in clips:
        n = clip.size(0)
        if n < max_players:
            pad = torch.zeros(max_players - n, *clip.shape[1:])
            clip = torch.cat([clip, pad], dim=0)
        padded.append(clip)

    labels = torch.stack(labels)[:, -1, :]  # take last frame label
    return torch.stack(padded), labels
