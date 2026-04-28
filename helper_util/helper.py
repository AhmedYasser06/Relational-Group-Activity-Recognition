import os
import yaml
import torch
import pickle


class Config:
    def __init__(self, d):
        self.model = d.get('model', {})
        self.training = d.get('training', {})
        self.data = d.get('data', {})
        self.experiment = d.get('experiment', {})

    def __repr__(self):
        return f"Config(model={self.model}, training={self.training}, data={self.data}, experiment={self.experiment})"


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return Config(yaml.safe_load(f))


def save_checkpoint(model, optimizer, epoch, val_acc, config, exp_dir, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config,
        'exp_dir': exp_dir,
    }

    path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch}.pkl')
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

    if is_best:
        best_path = os.path.join(exp_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        torch.save(checkpoint, checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is None:
        return model

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # move optimizer tensors to the right device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return (
        model,
        optimizer,
        checkpoint.get('config'),
        checkpoint.get('exp_dir'),
        checkpoint['epoch'] + 1,
    )
