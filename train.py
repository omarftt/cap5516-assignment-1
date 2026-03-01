import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

WARMUP_EPOCHS = 5

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss/total, correct/total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss/total, correct/total


def train(model, loaders, args, device, checkpoint_path, writer: SummaryWriter):
    # Load loss function with weight loss depend of our experiment
    if args.use_weighted_loss:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=torch.tensor([3.0, 1.0]).to(device))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Load scheduler with warmup depend of our experiment
    if args.use_warmup:
        warmup  = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6 / args.lr, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print("Init training")
    # Training each epoch
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        print(f"Epoch {epoch:3d}/{args.epochs}, "
              f"Train loss: {train_loss:.4f}  Acc: {train_acc:.4f}, "
              f"Val loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, checkpoint_path)
            print(f"Best model saved")

    print(f"Best Val Acc: {best_val_acc:.4f}")
    return history