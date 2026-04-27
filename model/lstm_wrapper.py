import torch
import torch.nn as nn
import torch.optim as optim
from .lstm_model import LSTMModel

def train_lstm(X_train, y_train, input_size, epochs=10):
    model = LSTMModel(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)

        loss.backward()
        optimizer.step()

    return model
