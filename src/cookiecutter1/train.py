import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import random_split
import statistics
app = typer.Typer()

def train(lr: float = 1e-3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"Learning rate: {lr}")

    # Initialize the model and dataset
    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()

    # Split into training and validation sets
    train_size = int(0.8 * len(train_set))  # 80% for training
    val_size = len(train_set) - train_size  # Remaining 20% for validation
    train_data, val_data = random_split(train_set, [train_size, val_size])

    # Data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training setup
    epochs = 5
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

    # Save the trained model
    torch.save(model.state_dict(), "models/model_checkpoint.pth")

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
    plt.show()

if __name__ == "__main__":
    typer.run(train)