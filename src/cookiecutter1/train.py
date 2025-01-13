import sys
print("Using Python version:", sys.version)
import torch
import typer
from .data import corrupt_mnist
from .model import MyAwesomeModel
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import random_split
import statistics
import wandb

app = typer.Typer()


def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"Learning rate: {lr}")
    wandb.login()
    run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
    },
    )
    # Initialize the model and dataset
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()

    # Split into training and validation sets
    train_size = int(0.8 * len(train_set))  # 80% for training
    val_size = len(train_set) - train_size  # Remaining 20% for validation
    train_data, val_data = random_split(train_set, [train_size, val_size])
    # Data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    a = 1
    statistics = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        running_loss = 0
        model.train()  # Set model to training mode

        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        val_loss = 0
        accuracy = 0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for images, labels in valloader:
                log_ps = model(images)
                val_loss += criterion(log_ps, labels).item()
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Record statistics
        train_loss_avg = running_loss / len(trainloader)
        val_loss_avg = val_loss / len(valloader)
        val_accuracy_avg = accuracy / len(valloader)

        statistics["train_loss"].append(train_loss_avg)
        statistics["val_loss"].append(val_loss_avg)
        statistics["val_accuracy"].append(val_accuracy_avg)

        print(
            f"Epoch {epoch + 1}/{epochs}.. "
            f"Train loss: {train_loss_avg:.3f}.. "
            f"Val loss: {val_loss_avg:.3f}.. "
            f"Val accuracy: {val_accuracy_avg:.3f}"
        )
        wandb.log({"accuracy": val_accuracy_avg, "loss": val_loss_avg})

    # Save the trained model
    artifact = wandb.Artifact("trained-model", type="model")

    torch.save(model.state_dict(), "models/model_checkpoint.pth")

    artifact.add_file("models/model_checkpoint.pth")
    artifact.save()

    # Plot training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], label="Train Loss")
    axs[0].plot(statistics["val_loss"], label="Validation Loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(statistics["val_accuracy"], label="Validation Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    fig.savefig("reports/figures/training_statistics.png")
    wandb.log({"training_statistics": wandb.Image("reports/figures/training_statistics.png")})


if __name__ == "__main__":
    typer.run(train)
