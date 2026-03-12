import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.edulite_model import EduLiteLLM


def main():

    # load dataset
    inputs = np.load("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/inputs.npy")
    targets = np.load("C:/Users/Dell/OneDrive/Desktop/EduliteLLM/targets.npy")

    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    print("Dataset shape:", inputs.shape)

    dataset = TensorDataset(inputs, targets)
    batch_size = 64

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  
    )

    vocab_size = 20000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    model = EduLiteLLM(vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5

    for epoch in range(epochs):

        total_loss = 0

        for step, (x, y) in enumerate(dataloader):

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = loss_fn(outputs.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 500 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")

        print(f"Epoch {epoch+1} completed. Total Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "C:/Users/Dell/OneDrive/Desktop/EduliteLLM/edulite_model.pt")

    print("Model trained and saved!")


if __name__ == "__main__":
    main()