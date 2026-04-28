import os
import torch


def get_sampler_weights(dataset, cache_path='sampler_weights.pth'):
    if os.path.exists(cache_path):
        saved = torch.load(cache_path)
        return saved['samples_weights'], saved['class_weights']

    labels = [dataset[i][1].argmax().item() for i in range(len(dataset))]
    labels_tensor = torch.tensor(labels)

    class_counts = torch.bincount(labels_tensor)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts.float())
    class_weights = class_weights / class_weights.sum()

    samples_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float)

    torch.save({'samples_weights': samples_weights, 'class_weights': class_weights}, cache_path)

    return samples_weights, class_weights
