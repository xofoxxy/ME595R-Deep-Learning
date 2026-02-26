import torch
from torch import nn
import torchdiffeq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class ODEFunc(nn.Module):
    def __init__(self, depth, width, dim):
        super(ODEFunc, self).__init__()
        layers = []
        layers.append(nn.Linear(dim+1, width))
        layers.append(nn.SiLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(width, dim))

        self.net = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def train(self, data, testing=True, epochs_per_chunk=50, chunk_size=4, lr=1e-3):
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0) + 1e-8
        data_normalized = (data - data_mean) / data_std

        # store stats so test() can reuse them
        self.data_mean = data_mean.detach()
        self.data_std = data_std.detach()

        # (optional) set lr like their ADAMW(lr) behavior
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        N = data_normalized.shape[0]
        t_full = torch.arange(N, dtype=data.dtype, device=data.device) / 12.0
        all_losses = []
        k_values = list(range(chunk_size, N + 1, chunk_size))

        for k in k_values:
            y_k = data_normalized[:k]
            t_k = t_full[:k]
            y0 = y_k[0]

            for epoch in range(epochs_per_chunk):
                self.optimizer.zero_grad()

                pred = torchdiffeq.odeint(self, y0, t_k)  # shape [k, dim]
                loss = torch.mean((pred - y_k) ** 2)

                loss.backward()
                self.optimizer.step()

                all_losses.append(loss.item())

                if epoch % 10 == 0:
                    print(f"k={k} | epoch {epoch}/{epochs_per_chunk} | loss {loss.item()}")

            if testing:
                with torch.no_grad():
                    pred = torchdiffeq.odeint(self, y0, t_k)
                    print(f"Fit loss up to k={k}: {torch.mean((pred - y_k) ** 2).item()}")
                self.test(data, training_data=data_normalized[:k])

        plt.plot(all_losses)
        plt.xlabel("Update step (epochs across rounds)")
        plt.ylabel("MSE")
        plt.title("Training Loss (Growing Prefix Rounds)")
        plt.show()

    def test(self, data, training_data):

        with torch.no_grad():
            if hasattr(self, "data_mean") and hasattr(self, "data_std"):
                data_mean = self.data_mean.to(device=data.device, dtype=data.dtype)
                data_std = self.data_std.to(device=data.device, dtype=data.dtype) + 1e-8
            else:
                data_mean = training_data.mean(dim=0).to(device=data.device, dtype=data.dtype)
                data_std = training_data.std(dim=0).to(device=data.device, dtype=data.dtype) + 1e-8

            data_normalized = (data - data_mean) / data_std

            t = torch.arange(data_normalized.shape[0], dtype=data.dtype, device=data.device)/12
            y0 = data_normalized[0]

            pred_normalized = torchdiffeq.odeint(self, y0, t)
            pred = pred_normalized * data_std + data_mean

            train_len = training_data.shape[0]

            overall_mse = torch.mean((pred_normalized - data_normalized) ** 2).item()
            test_mse = torch.mean((pred_normalized[train_len:] - data_normalized[train_len:]) ** 2).item()

            print("Overall MSE:", overall_mse)
            print("Test-region MSE:", test_mse)

            pred_np = pred.detach().cpu().numpy()
            data_np = data.detach().cpu().numpy()

            x = np.arange(len(data_np))

            fig, axes = plt.subplots(data_np.shape[1], 1, figsize=(8, 8), sharex=True)
            if data_np.shape[1] == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                ax.scatter(x, data_np[:, i], label="Data")
                ax.plot(x, pred_np[:, i], label="Model")
                ax.legend()

            axes[-1].set_xlabel("Timestep")
            plt.tight_layout()
            plt.show()

    def forward(self, t, x):
        return self.net(torch.cat([x, t.view(1)], dim=0))

def import_data():
    data1 = pd.read_csv("archive(1)/DailyDelhiClimateTest.csv")
    data2 = pd.read_csv("archive(1)/DailyDelhiClimateTrain.csv")
    data = pd.concat([data1, data2], ignore_index=True)

    data['date'] = pd.to_datetime(data['date'])
    month_averaged_data = data.groupby([data['date'].dt.year, data['date'].dt.month]).mean()

    data = month_averaged_data

    data = data.drop(columns=['date'])
    data = data.reset_index(drop=True)

    test_data = data.iloc[20:]
    training_data = data.iloc[:20]

    # print(data)

    return training_data, test_data


if __name__ == '__main__':
    training_data, test_data = import_data()

    all_the_data = pd.concat([training_data, test_data], ignore_index=True)

    model = ODEFunc(depth=4, width=128, dim=training_data.shape[1])
    print(training_data.values)

    model.train(torch.tensor(all_the_data.values, dtype=torch.float32), testing=True)
    model.test(torch.tensor(all_the_data.values, dtype=torch.float32),
               torch.tensor(all_the_data.values, dtype=torch.float32))
