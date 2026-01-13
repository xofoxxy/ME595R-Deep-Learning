import pandas as pd
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


def clean_data(file_name):
    # Read space-delimited file with custom whitespace
    cleaned_df = pd.read_csv(file_name, delim_whitespace=True, header=None, names=[
        "mpg", "cylinders", "displacement", "horsepower", "weight", 
        "acceleration", "model year", "origin", "name"
    ])
    
    # Handle the '?' values in horsepower
    cleaned_df['horsepower'] = pd.to_numeric(cleaned_df['horsepower'], errors='coerce')
    
    # Drop rows with any NA values
    cleaned_df = cleaned_df.dropna()
    
    # Convert all columns to numeric (except name)
    numeric_columns = cleaned_df.columns.drop('name')
    for col in numeric_columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col])
    
    # Normalize numeric columns
    stats = {}

    for col in numeric_columns:
        mean = cleaned_df[col].mean()
        std = cleaned_df[col].std()

        stats[col] = (mean, std)

        cleaned_df[col] = (cleaned_df[col] - mean) / std
    
    return cleaned_df, stats


def seperate_data(df):
    n = len(df)
    training_n = int(n * 0.8)

    df = df.sample(frac=1).reset_index(drop=True)
    training = df[:training_n]
    testing = df[training_n:]
    return training, testing


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)  # Because my data is already a vector, I don't need to flatten


class CarDataset(Dataset):
    def __init__(self, df):
        y = df["mpg"].values
        X = df.drop(["mpg", "name"], axis=1).values
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_model(model, training_data, test_data, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in training_data:
            y_pred = model(batch_x)

            loss = criterion(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            plt.scatter(epoch, loss.item(), c="red", label="Training MSE Loss")

        for t_batch_x, t_batch_y in test_data:
            t_y_pred = model(t_batch_x)
            t_loss = criterion(t_y_pred, t_batch_y)

            plt.scatter(epoch, t_loss.item(), c="blue", label="Test MSE Loss")

        if epoch % 10 == 0:
            print(f"Average Epoch {epoch}: Loss {total_loss / len(training_data)}")


def eval_model(model, test_data):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_x, batch_y in test_data:
            y_pred = model(batch_x)
            loss = nn.MSELoss()(y_pred, batch_y)
            test_loss += loss.item()

        print(f"Test Loss: {test_loss / len(test_data)}")
        return test_loss / len(test_loader)


def build_graph(test_loss):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title("Training and Test Losses over Epochs")

    plt.annotate(f"Average loss after epochs: {test_loss}",
                 xy=(320, 240),
                 color="black",
                 bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

    plt.savefig("loss_over_epochs.png")
    plt.show()


if __name__ == "__main__":
    data_filepath = "auto-mpg.data"
    df, stats = clean_data(data_filepath) # stats comes in the form of (mean, std)

    training_df, testing_df = seperate_data(df)

    train_dataset = CarDataset(training_df)
    test_dataset = CarDataset(testing_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = NeuralNetwork()
    train_model(model, train_loader, test_loader, epochs=100)

    end_loss = eval_model(model, test_loader) * stats["mpg"][1]
    print(f"Final Test Loss: {end_loss}")

    build_graph(end_loss)