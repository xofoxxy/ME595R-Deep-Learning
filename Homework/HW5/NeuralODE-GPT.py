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
        layers.append(nn.Linear(dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, dim))

        self.net = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def train(self, data, train_len=20, epochs=2000):

        train_raw = data[:train_len]
        full_raw = data

        mean = train_raw.mean(dim=0)
        std = train_raw.std(dim=0) + 1e-8
        train = (train_raw - mean) / std
        full = (full_raw - mean) / std

        t_train = torch.arange(train.shape[0], dtype=data.dtype, device=data.device) / 12.0
        t_full = torch.arange(full.shape[0], dtype=data.dtype, device=data.device) / 12.0

        y0 = train[0]

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            pred_train = torchdiffeq.odeint(self, y0, t_train)
            train_loss = torch.mean((pred_train - train) ** 2)

            train_loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                pred_full = torchdiffeq.odeint(self, y0, t_full)
                test_loss = torch.mean(
                    (pred_full[train_len:] - full[train_len:]) ** 2
                )

            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"train MSE {train_loss.item():.6f} | "
                    f"test MSE {test_loss.item():.6f}"
                )

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Training and Test Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def test(self, data, training_data):

        with torch.no_grad():
            data_mean = data.mean(dim=0)
            data_std = data.std(dim=0)
            data_normalized = (data - data_mean) / (data_std + 1e-8)

            # Same "months" timebase as train()
            t = (torch.arange(data_normalized.shape[0], dtype=data_normalized.dtype, device=data_normalized.device) / 12.0)

            y0 = data_normalized[0]
            pred_normalized = torchdiffeq.odeint(self, y0, t)

            pred = pred_normalized * (data_std + 1e-8) + data_mean

            print(f"Test loss: {torch.mean((pred_normalized - data_normalized) ** 2).item()}")

            pred_np = pred.detach().cpu().numpy()
            data_np = data.detach().cpu().numpy()

            train_len = training_data.shape[0]
            test_start = train_len

            x = np.arange(data_np.shape[0])
            use_dates = False
            dates = None
            try:
                dates = pd.date_range(start="2013-01-01", periods=len(x), freq="MS").to_pydatetime()
                use_dates = True
            except Exception:
                use_dates = False
                dates = x

            feature_names = ["Mean temperature", "Humidity", "Wind speed", "Mean pressure"]
            ylabels = ["Celsius", "g/m$^3$ of water", "km/h", "hPa"]

            fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
            fig.subplots_adjust(hspace=0.60)

            cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

            for i, ax in enumerate(axes):
                c = cycle[i % len(cycle)]

                ax.scatter(
                    dates, data_np[:, i],
                    s=46, facecolor=c, edgecolor="black", linewidth=1.2,
                    label="Observations", zorder=3
                )

                ax.plot(
                    dates[:train_len], pred_np[:train_len, i],
                    color=c, linewidth=2.4, linestyle="-",
                    label="Fit", zorder=2
                )

                if test_start < len(x):
                    ax.plot(
                        dates[test_start:], pred_np[test_start:, i],
                        color=c, linewidth=2.4, linestyle="--",
                        label="Forecast", zorder=2
                    )

                ax.set_title(feature_names[i], fontsize=14, pad=10)
                ax.set_ylabel(ylabels[i])
                ax.grid(True, alpha=0.25)

                if i == 3:
                    ax.legend(loc="upper right", frameon=True)

            if use_dates:
                axes[-1].xaxis.set_major_locator(mdates.YearLocator())
                axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            axes[-1].set_xlabel("Time")

            plt.tight_layout()
            plt.show()

    def forward(self, t, x):
        return self.net(x)


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

    return training_data, test_data


if __name__ == '__main__':
    training_data, test_data = import_data()

    all_the_data = pd.concat([training_data, test_data], ignore_index=True)

    model = ODEFunc(depth=4, width=128, dim=training_data.shape[1])
    print(training_data.values)

    model.train(torch.tensor(training_data.values, dtype=torch.float32))
    model.test(torch.tensor(all_the_data.values, dtype=torch.float32),
               torch.tensor(training_data.values, dtype=torch.float32))
