import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from models.net import ArrhythmiaNet


def load_data(client_id):
    """Load and preprocess the Arrhythmia dataset for a specific client."""
    df = pd.read_csv('/app/data/arrhythmia.csv')
    
    X = df[['MLII', 'V5']].values
    
    bins = np.linspace(df['MLII'].min(), df['MLII'].max(), 5)
    y = np.digitize(df['MLII'], bins) - 1  # Subtract 1 to make labels start from 0
    
    print(f"Unique labels: {np.unique(y)}")
   
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_clients = 3
    client_idx = client_id - 1
    X_split = np.array_split(X, n_clients)[client_idx]
    y_split = np.array_split(y, n_clients)[client_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_split, y_split, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        ),
        batch_size=32,
        shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        ),
        batch_size=32,
        shuffle=False
    )
    
    return train_loader, test_loader



class ArrhythmiaClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        self.set_parameters(parameters)
        
        # Local training loop
        for epoch in range(5):  # Local epochs
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} "
                          f"({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test dataset."""
        self.set_parameters(parameters)
        
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Calculate average loss and accuracy
        num_samples = len(self.test_loader.dataset)
        loss /= len(self.test_loader)
        accuracy = correct / num_samples
        
        print(f"Test set: Average loss: {loss:.4f}, Accuracy: {correct}/{num_samples} ({100.0 * accuracy:.2f}%)")
        
        return loss, num_samples, {"accuracy": accuracy}

import time


def main():
    client_id = int(os.getenv("CLIENT_ID", 1))
    server_address = os.getenv("SERVER_ADDRESS", "server:8080")
    
    time.sleep(10)
    
    print(f"Starting client {client_id} with server address {server_address}")
 
    train_loader, test_loader = load_data(client_id)

    model = ArrhythmiaNet()
  
    client = ArrhythmiaClient(model, train_loader, test_loader)
    fl.client.start_numpy_client(
        server_address=server_address,  
        client=client
    )


if __name__ == "__main__":
    main()
