import torch
import torch.nn as nn
from cookiecutter1.model import MyAwesomeModel
from cookiecutter1.data import corrupt_mnist
import typer

app = typer.Typer()

def evaluate() -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    path = "models/model_checkpoint.pth"

    model = MyAwesomeModel()

    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    _, test_set = corrupt_mnist()

    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    criterion = nn.NLLLoss()

    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            # Forward pass
            log_ps = model(images)

            loss = criterion(log_ps, labels)
            test_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            correct += (top_class.view(-1) == labels).sum().item()
            total += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = test_loss / len(testloader)
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.3f}")
    print(f"Test Accuracy: {accuracy:.3%}")


if __name__ == "__main__":
    typer.run(evaluate)
