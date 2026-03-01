import matplotlib.pyplot as plt
import os


def plot_loss_curves(history: dict, plots_dir: str, run_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"],   label="validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves"); axes[1].legend(); axes[1].grid(True)

    plt.suptitle(run_name)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{run_name}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()