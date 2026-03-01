import argparse

DATA_DIR       = "data"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "logs"
RESULTS_DIR    = "results"
PLOTS_DIR      = "plots"

IMG_SIZE    = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def get_args():
    parser = argparse.ArgumentParser(description="vit")

    parser.add_argument("--run_name", type=str, required=True, help="Name for this run")
    parser.add_argument("--pretrained", action="store_true", help="Activate to use pretrained or not")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Experiment flags
    parser.add_argument("--use_warmup", action="store_true", help="warmup before cosine decay")
    parser.add_argument("--use_weighted_loss", action="store_true", help="use for experiment with different weights loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor (0.0 to disable)")
    parser.add_argument("--use_random_erasing", action="store_true", help="Randomly masks patches during training")

    # For eval
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Choose mode")
    parser.add_argument("--gradcam", action="store_true", help="Add grad-cam to failure case plots")

    return parser.parse_args()