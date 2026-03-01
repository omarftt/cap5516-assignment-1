import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
import numpy as np
import os

from utils.gradcam import ViTGradCAM, overlay_cam
from utils.dataset import MEAN, STD

inv_norm = transforms.Normalize(
    mean=[-m / s for m, s in zip(MEAN, STD)],
    std=[1 / s for s in STD],
)

@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs, all_imgs = [], [], [], []

    for imgs, labels in loader:
        probs = torch.softmax(model(imgs.to(device)), dim=1).cpu()
        all_preds.append(probs.argmax(1))
        all_labels.append(labels)
        all_probs.append(probs)
        all_imgs.append(imgs)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    all_imgs = torch.cat(all_imgs)

    overall_acc = (all_preds == all_labels).float().mean().item()
    per_class_acc = {
        name: (all_preds[all_labels == i] == i).float().mean().item()
        for i, name in enumerate(class_names)
    }
    return overall_acc, per_class_acc, all_preds, all_labels, all_imgs, all_probs


def save_results(overall_acc, per_class_acc, all_preds, all_labels, class_names, results_dir, run_name):
    report = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=class_names, digits=4)

    print(f"\nOverall Accuracy: {overall_acc}")
    for cls, acc in per_class_acc.items():
        print(f"{cls}:{acc}")
    print(report)

    # Confusion matrix
    cm  = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{run_name}_confusion_matrix.png"), dpi=150)
    plt.close()


def save_failure_cases(model, all_imgs, all_preds, all_labels, all_probs,
                       class_names, plots_dir, run_name, num_cases=6,
                       device=torch.device("cpu"), use_gradcam=False):
    wrong = (all_preds != all_labels).nonzero(as_tuple=True)[0][:num_cases]

    if len(wrong) == 0:
        print("No failure cases")
        return

    cols = 2 if use_gradcam else 1
    fig, axes = plt.subplots(len(wrong), cols, figsize=(cols * 3, len(wrong) * 3))
    axes = np.array(axes).reshape(len(wrong), cols)   # always (n, cols)

    gcam = ViTGradCAM(model) if use_gradcam else None

    for row, idx in enumerate(wrong):
        img_t    = all_imgs[idx]
        img_disp = inv_norm(img_t).permute(1, 2, 0).clamp(0, 1).numpy()
        true_lbl = class_names[all_labels[idx].item()]
        pred_lbl = class_names[all_preds[idx].item()]
        conf     = all_probs[idx][all_preds[idx]].item()

        axes[row, 0].imshow(img_disp)
        axes[row, 0].set_title(f"True: {true_lbl}  Pred: {pred_lbl} ({conf:.2f})", fontsize=8, color="red")
        axes[row, 0].axis("off")

        if use_gradcam:
            cam = gcam(img_t.unsqueeze(0).to(device), class_idx=all_preds[idx].item())
            axes[row, 1].imshow(overlay_cam((img_disp * 255).astype(np.uint8), cam))
            axes[row, 1].set_title("Grad-CAM", fontsize=8)
            axes[row, 1].axis("off")

    if gcam:
        gcam.remove_hooks()

    plt.suptitle(f"Failure Cases", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{run_name}_failure_cases.png"), dpi=150)
    plt.close()