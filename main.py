import torch
import torch.nn as nn
import torch.optim as optim

from models import resnet18
from utils import get_dataloaders, train_one_epoch, evaluate, plot_curves, plot_confusion_matrix


def main():
    # Hyperparameters
    epochs = 20
    lr = 0.1
    batch_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    trainloader, testloader = get_dataloaders(batch_size)

    # Model
    model = resnet18(num_classes=10).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # Training Loop
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc, preds, labels = evaluate(model, testloader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%")

    # Plots
    plot_curves(train_losses, test_losses, train_accs, test_accs)
    plot_confusion_matrix(labels, preds, class_names=testloader.dataset.classes)


if __name__ == "__main__":
    main()
