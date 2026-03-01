import torch
from torch.utils.tensorboard import SummaryWriter
from utils.config import get_args, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, PLOTS_DIR, CLASS_NAMES
from utils.dataset import get_dataloaders
from utils.model import ViTTiny
from utils.utils import plot_loss_curves
from train import train
from evaluate import evaluate, save_results, save_failure_cases
import os

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Run: {args.run_name}")
    print(f"Pretrained: {args.pretrained}\n")

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{args.run_name}_best.pth")

    # Load data
    loaders = get_dataloaders(args.batch_size)
    print(f"Train: {len(loaders['train'].dataset)}\n"
          f"Val: {len(loaders['val'].dataset)}\n"
          f"Test: {len(loaders['test'].dataset)}\n")

    # Load model
    model = ViTTiny(pretrained=args.pretrained).to(device)
    print(f"ViTTiny loaded")

    # Training phase
    if args.mode == "train":
        writer  = SummaryWriter(log_dir=os.path.join(LOG_DIR, args.run_name))
        history = train(model, loaders, args, device, checkpoint_path, writer)
        writer.close()
        plot_loss_curves(history, PLOTS_DIR, args.run_name)

    # Testing phase
    if args.mode == "test":
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print("Checkpoint loaded")
    
        overall_acc, per_class_acc, all_preds, all_labels, all_imgs, all_probs = evaluate(model, loaders["test"], device, CLASS_NAMES)

        save_results(overall_acc, per_class_acc, all_preds, all_labels,
                    CLASS_NAMES, RESULTS_DIR, args.run_name)

        save_failure_cases(
            model, all_imgs, all_preds, all_labels, all_probs,
            CLASS_NAMES, PLOTS_DIR, args.run_name,
            num_cases=4, device=device, use_gradcam=args.gradcam,
        )


if __name__ == "__main__":
    main()