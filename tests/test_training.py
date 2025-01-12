from src.cookiecutter1.train import train
from src.cookiecutter1.model import MyAwesomeModel
from src.cookiecutter1.data import corrupt_mnist
from torch.utils.data import DataLoader
import torch
import pytest

def test_overfitting(batch_size=32, learning_rate=0.01):
    train_set, _ = corrupt_mnist()
    model = MyAwesomeModel()
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    initial_loss = None
    final_loss = None

    # Run more training steps with a higher learning rate
    model.train()
    try:
        for i, (images, labels) in enumerate(trainloader):
            if i == 30:  # Train for 10 batches instead of 3
                break
            optimizer.zero_grad()
            log_ps = model(images)

            # Ensure output shape is correct
            assert log_ps.shape[1] == 10, "Model output does not match the number of classes."

            loss = criterion(log_ps, labels)
            if i == 0:  # Record initial loss
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
            print(f"Batch {i}: Loss = {loss.item()}")

        # Check if loss decreased or stabilized
        assert initial_loss is not None and final_loss is not None, "Loss values were not recorded."
        assert initial_loss > final_loss, f"Loss did not decrease: Initial = {initial_loss}, Final = {final_loss}"

        print(f"Test passed: Initial loss = {initial_loss}, Final loss = {final_loss}")

    except Exception as e:
        pytest.fail(f"Overfitting test failed with error: {e}")
