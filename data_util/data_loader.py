import cv2
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from .boxinfo import BoxInfo


person_activity_classes = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_labels = {c.lower(): i for i, c in enumerate(person_activity_classes)}

group_activity_classes = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {c: i for i, c in enumerate(group_activity_classes)}

activities_labels = {
    "person": person_activity_labels,
    "group": group_activity_labels,
}


def load_annot(annot_path):
    with open(annot_path, 'rb') as f:
        return pickle.load(f)


def read_frame(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    return frame


def box_center_x(box):
    x1, _, x2, _ = box
    return (x1 + x2) / 2


def crop_person(frame, box, transform=None):
    x1, y1, x2, y2 = box.box
    crop = frame[y1:y2, x1:x2]
    if transform:
        crop = transform(image=crop)['image']
    return crop


# Dataset 1: person-level activity

class PersonActivityDataset(Dataset):
    """
    Returns individual person crops with per-person action labels.
    Set seq=True to get full clip sequences instead of single frames.
    """
    def __init__(self, videos_path, annot_path, split, labels, seq=False, only_tar=False, transform=None):
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.seq = seq
        self.only_tar = only_tar
        self.labels = labels

        videos_annot = load_annot(annot_path)

        self.samples = []
        for clip_id in split:
            clip_data = videos_annot[str(clip_id)]
            for clip_dir, clip_info in clip_data.items():
                frames_data = clip_info['frame_boxes_dct']

                if seq:
                    frames = [
                        {'frame_id': fid, 'boxes': boxes}
                        for fid, boxes in frames_data.items()
                    ]
                    if frames:
                        self.samples.append({
                            'type': 'seq',
                            'clip_id': clip_id,
                            'clip_dir': clip_dir,
                            'frames': frames,
                        })
                else:
                    for frame_id, boxes in frames_data.items():
                        if only_tar and str(frame_id) != str(clip_dir):
                            continue
                        for box in boxes:
                            self.samples.append({
                                'type': 'single',
                                'clip_id': clip_id,
                                'clip_dir': clip_dir,
                                'frame_id': frame_id,
                                'box': box,
                            })

    def __len__(self):
        return len(self.samples)

    def _get_frame_path(self, clip_id, clip_dir, frame_id):
        return self.videos_path / str(clip_id) / str(clip_dir) / f"{frame_id}.jpg"

    def _make_label(self, category):
        label = np.zeros(len(self.labels))
        label[self.labels[category]] = 1
        return label

    def __getitem__(self, idx):
        s = self.samples[idx]

        if s['type'] == 'single':
            frame = read_frame(self._get_frame_path(s['clip_id'], s['clip_dir'], s['frame_id']))
            crop = crop_person(frame, s['box'], self.transform)
            label = self._make_label(s['box'].category)
            return crop, torch.from_numpy(label).float()

        # sequence mode (num_people, num_frames, C, H, W)
        all_frame_crops = []
        all_frame_labels = []

        for fd in s['frames']:
            frame = read_frame(self._get_frame_path(s['clip_id'], s['clip_dir'], fd['frame_id']))
            crops = [crop_person(frame, b, self.transform) for b in fd['boxes']]
            labels = [self._make_label(b.category) for b in fd['boxes']]

            if crops:
                all_frame_crops.append(np.stack(crops))
                all_frame_labels.append(np.stack(labels))

        crops_tensor = np.transpose(np.stack(all_frame_crops), (1, 0, 2, 3, 4))
        labels_tensor = np.transpose(np.stack(all_frame_labels), (1, 0, 2))

        return torch.from_numpy(crops_tensor), torch.from_numpy(labels_tensor).float()


# Dataset 2: group-level activity

class GroupActivityDataset(Dataset):
    """
    Returns all player crops for a frame/clip plus a group-level label.
    Set seq=True to get the full 9-frame clip.
    Set sort=True to sort players by their x-position (left to right).
    """
    def __init__(self, videos_path, annot_path, split, labels, seq=False, sort=False, only_tar=False, transform=None):
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.seq = seq
        self.sort = sort
        self.labels = labels

        videos_annot = load_annot(annot_path)

        self.samples = []
        for clip_id in split:
            clip_data = videos_annot[str(clip_id)]
            for clip_dir, clip_info in clip_data.items():
                frames_data = clip_info['frame_boxes_dct']
                category = clip_info['category']

                if seq:
                    frames = [
                        (f"{videos_path}/{clip_id}/{clip_dir}/{fid}.jpg", boxes)
                        for fid, boxes in frames_data.items()
                    ]
                    self.samples.append({'type': 'seq', 'frames': frames, 'category': category})
                else:
                    for frame_id, boxes in frames_data.items():
                        if only_tar and str(frame_id) != str(clip_dir):
                            continue
                        frame_path = f"{videos_path}/{clip_id}/{clip_dir}/{frame_id}.jpg"
                        self.samples.append({'type': 'single', 'frame_path': frame_path, 'boxes': boxes, 'category': category})

    def __len__(self):
        return len(self.samples)

    def _group_label(self, category):
        label = torch.zeros(len(self.labels))
        label[self.labels[category]] = 1
        return label

    def _get_crops(self, frame, boxes):
        crops, centers = [], []
        for box in boxes:
            crops.append(crop_person(frame, box, self.transform))
            centers.append(box_center_x(box.box))

        if self.sort:
            crops = [c for _, c in sorted(zip(centers, crops), key=lambda p: p[0])]

        return torch.stack(crops)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = self._group_label(s['category'])

        if s['type'] == 'single':
            frame = read_frame(s['frame_path'])
            crops = self._get_crops(frame, s['boxes'])
            return crops, label

        # seq mode (num_people, num_frames, C, H, W)
        clip, labels = [], []
        for frame_path, boxes in s['frames']:
            frame = read_frame(frame_path)
            clip.append(self._get_crops(frame, boxes))
            labels.append(label)

        clip = torch.stack(clip).permute(1, 0, 2, 3, 4)
        labels = torch.stack(labels)
        return clip, labels


# Dataset 3: end-to-end (person + group labels)

class End2EndDataset(Dataset):
    """
    Returns clip crops, per-person labels, and group labels together.
    Used for end-to-end training with both supervision signals.
    Output shapes: clip (N, T, C, H, W), person_labels (N, T, num_person_cls), group_labels (T, num_group_cls)
    """
    def __init__(self, videos_path, annot_path, split, labels, transform=None):
        self.videos_path = Path(videos_path)
        self.transform = transform
        self.labels = labels

        videos_annot = load_annot(annot_path)

        self.samples = []
        for clip_id in split:
            clip_data = videos_annot[str(clip_id)]
            for clip_dir, clip_info in clip_data.items():
                category = clip_info['category']
                frames = [
                    (f"{videos_path}/{clip_id}/{clip_dir}/{fid}.jpg", boxes)
                    for fid, boxes in clip_info['frame_boxes_dct'].items()
                ]
                self.samples.append({'frames': frames, 'category': category})

    def __len__(self):
        return len(self.samples)

    def _person_label(self, category):
        label = torch.zeros(len(self.labels['person']))
        label[self.labels['person'][category]] = 1
        return label

    def _group_label(self, category):
        label = torch.zeros(len(self.labels['group']))
        label[self.labels['group'][category]] = 1
        return label

    def __getitem__(self, idx):
        s = self.samples[idx]
        group_label = self._group_label(s['category'])

        clip, group_labels, person_labels = [], [], []

        for frame_path, boxes in s['frames']:
            frame = read_frame(frame_path)

            crops, centers, p_labels = [], [], []
            for box in boxes:
                crops.append(crop_person(frame, box, self.transform))
                centers.append(box_center_x(box.box))
                p_labels.append(self._person_label(box.category))

            # sort by x position
            sorted_data = sorted(zip(centers, crops, p_labels), key=lambda x: x[0])
            crops = torch.stack([c for _, c, _ in sorted_data])
            p_labels = torch.stack([l for _, _, l in sorted_data])

            clip.append(crops)
            group_labels.append(group_label)
            person_labels.append(p_labels)

        clip = torch.stack(clip).permute(1, 0, 2, 3, 4)           # (N, T, C, H, W)
        group_labels = torch.stack(group_labels)                   # (T, num_group_cls)
        person_labels = torch.stack(person_labels).permute(1, 0, 2)  # (N, T, num_person_cls)

        return clip, person_labels, group_labels
