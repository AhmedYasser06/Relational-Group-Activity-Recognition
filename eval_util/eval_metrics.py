import logging
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def get_f1_score(y_true, y_pred, average='weighted', report=False):
    if report:
        print(classification_report(y_true, y_pred, zero_division=1))
        return
    score = f1_score(y_true, y_pred, average=average)
    print(f"F1 Score: {score:.4f}")
    return score


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ['d', '.2f'],
        ['Confusion Matrix (Counts)', 'Confusion Matrix (%)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved confusion matrix: {save_path}")

    plt.close(fig)
    return fig


def _run_inference(model, data_loader, device, criterion, end2end):
    y_true, y_pred = [], []
    total_loss = 0.0

    with torch.no_grad():
        if end2end:
            for inputs, person_labels, group_labels in data_loader:
                inputs = inputs.to(device)
                person_labels = person_labels.to(device)
                group_labels = group_labels.to(device)

                outputs = model(inputs)
                if criterion:
                    loss = criterion(outputs['group_output'], group_labels) + \
                           0.6 * criterion(outputs['person_output'], person_labels)
                    total_loss += loss.item()

                _, pred = outputs['group_output'].max(1)
                _, true = group_labels.max(1)
                y_pred.extend(pred.cpu().numpy())
                y_true.extend(true.cpu().numpy())
        else:
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                if criterion:
                    total_loss += criterion(outputs, targets).item()

                _, pred = outputs.max(1)
                _, true = targets.max(1)
                y_pred.extend(pred.cpu().numpy())
                y_true.extend(true.cpu().numpy())

    return y_true, y_pred, total_loss


def _build_log_message(prefix, accuracy, avg_loss, f1, y_true, y_pred, class_names, tta_count=None):
    tta_str = f" (TTA - {tta_count} transforms)" if tta_count else ""
    msg = (
        f"\n{'=' * 50}\n{prefix}{tta_str}\n{'=' * 50}\n"
        f"Accuracy : {accuracy:.2f}%\n"
    )
    if avg_loss is not None:
        msg += f"Average Loss: {avg_loss:.4f}\n"
    msg += f"F1 Score (Weighted): {f1:.4f}\n\nClassification Report:\n"
    msg += classification_report(y_true, y_pred, target_names=class_names)
    return msg


def model_eval(model, data_loader, criterion=None, path="", device=None,
               prefix="Test Set Report", class_names=None,
               log_path="evaluation.log", end2end=False):

    logging.basicConfig(filename=f"{path}/{log_path}", level=logging.INFO,
                        format='%(asctime)s - %(message)s', filemode='a')

    model.eval()
    y_true, y_pred, total_loss = _run_inference(model, data_loader, device, criterion, end2end)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    accuracy = report_dict.get('accuracy', 0) * 100
    avg_loss = total_loss / len(data_loader) if criterion else None
    f1 = f1_score(y_true, y_pred, average='weighted')

    msg = _build_log_message(prefix, accuracy, avg_loss, f1, y_true, y_pred, class_names)
    print(msg)
    logging.info(msg)

    if class_names:
        save_path = f"{path}/{prefix.replace(' ', '_')}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names, save_path)

    return {'accuracy': accuracy, 'avg_loss': avg_loss, 'f1_score': f1, 'classification_report': report_dict}


def model_eval_TTA(model, dataset, dataset_params, tta_transforms, criterion=None,
                   path="", device=None, prefix="Test Set Report", class_names=None,
                   end2end=False, log_path="TTA-evaluation.log"):

    logging.basicConfig(filename=f"{path}/{log_path}", level=logging.INFO,
                        format='%(asctime)s - %(message)s', filemode='a')

    model.eval()
    all_preds, all_targets = [], []

    for i, transform in enumerate(tta_transforms):
        print(f"TTA transform {i+1}/{len(tta_transforms)}")

        params = {**dataset_params, 'transform': transform}

        if end2end:
            ds = dataset(videos_path=params['videos_path'], annot_path=params['annot_path'],
                         split=params['split'], labels=params['labels'], transform=params['transform'])
        else:
            ds = dataset(videos_path=params['videos_path'], annot_path=params['annot_path'],
                         split=params['split'], labels=params['labels'], transform=params['transform'],
                         seq=params.get('seq', True), sort=params.get('sort', True),
                         only_tar=params.get('only_tar', False))

        loader = DataLoader(ds, batch_size=params.get('batch_size', 12), shuffle=False,
                            num_workers=params.get('num_workers', 1),
                            collate_fn=params.get('collate_fn', None),
                            pin_memory=params.get('pin_memory', True))

        preds, targets = [], []
        with torch.no_grad():
            if end2end:
                for inputs, person_labels, group_labels in loader:
                    out = model(inputs.to(device))
                    preds.extend(out['group_output'].cpu().tolist())
                    targets.extend(group_labels.cpu().tolist())
            else:
                for inputs, t in loader:
                    out = model(inputs.to(device))
                    preds.extend(out.cpu().tolist())
                    targets.extend(t.cpu().tolist())

        all_preds.append(torch.tensor(preds))
        all_targets.append(torch.tensor(targets))

    avg_preds = torch.mean(torch.stack(all_preds), dim=0)
    targets = all_targets[0].clone().detach()

    total_loss = 0.0
    if criterion:
        loss = criterion(avg_preds.to(device), targets.to(device))
        total_loss = loss.item() * targets.size(0)

    _, y_true = targets.max(1)
    _, y_pred = avg_preds.max(1)
    y_true, y_pred = y_true.numpy(), y_pred.numpy()

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    accuracy = report_dict.get('accuracy', 0) * 100
    avg_loss = total_loss / len(y_true) if criterion else None
    f1 = f1_score(y_true, y_pred, average='weighted')

    msg = _build_log_message(prefix, accuracy, avg_loss, f1, y_true, y_pred, class_names, tta_count=len(tta_transforms))
    print(msg)
    logging.info(msg)

    if class_names:
        save_path = f"{path}/{prefix.replace(' ', '_')}_TTA_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names, save_path)

    return {'accuracy': accuracy, 'avg_loss': avg_loss, 'f1_score': f1, 'classification_report': report_dict}
