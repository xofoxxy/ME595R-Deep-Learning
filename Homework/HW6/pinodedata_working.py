import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchdiffeq import odeint  # Neural ODE integrator


class DynamicsNetwork(nn.Module):
    """Learns dz/dt = f(z) in latent space"""
    def __init__(self, nz, width, depth):
        super(DynamicsNetwork, self).__init__()
        layers = [nn.Linear(nz, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(width, nz))
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        # t is required by odeint but not used (autonomous ODE)
        return self.net(z)


class Autoencoder(nn.Module):
    """Encoder: X → Z,  Decoder: Z → X"""
    def __init__(self, nx, nz, width, depth):
        super(Autoencoder, self).__init__()
        # Encoder
        enc_layers = [nn.Linear(nx, width), nn.SiLU()]
        for _ in range(depth - 1):
            enc_layers.append(nn.Linear(width, width))
            enc_layers.append(nn.SiLU())
        enc_layers.append(nn.Linear(width, nz))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (separate network, not just reversed layers)
        dec_layers = [nn.Linear(nz, width), nn.SiLU()]
        for _ in range(depth - 1):
            dec_layers.append(nn.Linear(width, width))
            dec_layers.append(nn.SiLU())
        dec_layers.append(nn.Linear(width, nx))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class PINODE(nn.Module):
    """
    Physics-Informed Neural ODE with Autoencoder
    Combines: Autoencoder (X ↔ Z) + Dynamics (dz/dt = f(z))
    """
    def __init__(self, nx, nz, width, depth):
        super(PINODE, self).__init__()
        self.autoencoder = Autoencoder(nx, nz, width, depth)
        self.dynamics = DynamicsNetwork(nz, width, depth)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def forward_dynamics(self, z0, t):
        """Integrate dynamics: given z(t0), predict z(t1), z(t2), ..."""
        # z0: (batch, nz), t: (nt,)
        # Returns: (nt, batch, nz)
        z_traj = odeint(self.dynamics, z0, t)
        return z_traj

    def predict_trajectory(self, x0, t):
        z0 = self.encode(x0)  # (batch, nz)
        z_traj = self.forward_dynamics(z0, t)  # (nt, batch, nz)
        # Decode each time step
        nt, batch, nz = z_traj.shape
        z_flat = z_traj.reshape(-1, nz)  # (nt*batch, nz)
        x_flat = self.decode(z_flat)  # (nt*batch, nx)
        x_traj = x_flat.reshape(nt, batch, -1)  # (nt, batch, nx)
        return x_traj.permute(1, 0, 2)  # (batch, nt, nx)


def train(model, Xtrain, t_train, Xcol, fcol, num_epochs, lr, Xtest, t_test):
    """
    Train the PINODE model with 3 loss components:
    1. Reconstruction loss (autoencoder)
    2. Dynamics loss (trajectory prediction)
    3. Physics loss (collocation points)
    """
    alpha_recon = 1.0   # weight for reconstruction loss
    alpha_dyn = 2.0     # weight for dynamics loss
    alpha_phys = 10    # weight for physics loss

    # Convert to torch tensors
    Xtrain_t = torch.tensor(Xtrain, dtype=torch.float32)
    t_t = torch.tensor(t_train, dtype=torch.float32)
    Xcol_t = torch.tensor(Xcol, dtype=torch.float32)
    fcol_t = torch.tensor(fcol, dtype=torch.float32)

    # ONE optimizer for ALL parameters (both networks)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses_per_epoch = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # === Loss 1: Reconstruction (autoencoder quality) ===
        loss_recon = reconstructionLoss(model, Xtrain_t)

        # === Loss 2: Dynamics (trajectory prediction) ===
        loss_dyn = dynamicsLoss(model, Xtrain_t, t_t)

        # === Loss 3: Physics (collocation points) ===
        loss_phys = physicsLoss(model, Xcol_t, fcol_t)

        # Total loss
        loss = alpha_recon * loss_recon + alpha_dyn * loss_dyn + alpha_phys * loss_phys
        loss.backward()
        optimizer.step()

        test_loss = compute_test_error(model, Xtest, t_test)

        losses = {
            "total": loss.item(),
            "reconstruction": loss_recon.item(),
            "dynamics": loss_dyn.item(),
            "physics": loss_phys.item(),
            "test": test_loss
        }
        losses_per_epoch.append(losses)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Total: {loss.item():.4e} | "
                  f"Recon: {loss_recon.item():.4e} | Dyn: {loss_dyn.item():.4e} | Phys: {loss_phys.item():.4e}")

    return losses_per_epoch


def reconstructionLoss(model, X):
    """
    Autoencoder reconstruction loss: ||X - decode(encode(X))||²
    Ensures the autoencoder can compress and reconstruct data.
    """
    # X shape: (ntraj, nt, nx) - flatten to (ntraj*nt, nx)
    X_flat = X.reshape(-1, X.shape[-1])
    X_recon = model.autoencoder(X_flat)
    return torch.mean((X_flat - X_recon) ** 2)


def dynamicsLoss(model, X, t):
    """
    Dynamics loss: Compare predicted trajectories to true trajectories.
    - Encode X(t0) to get z0
    - Integrate dynamics to get z(t)
    - Decode to get X_pred(t)
    - Compare to true X(t)
    """
    # X shape: (ntraj, nt, nx)
    X0 = X[:, 0, :]  # Initial conditions (ntraj, nx)
    X_pred = model.predict_trajectory(X0, t)  # (ntraj, nt, nx)
    return torch.mean((X - X_pred) ** 2)


def physicsLoss(model, Xcol, fcol):
    # Xcol: (B, nx), fcol: (B, nx)
    Xcol = Xcol.requires_grad_(True)

    z = model.encode(Xcol)              # (B, nz)
    dz_dt_pred = model.dynamics(None, z) # (B, nz)

    # Compute dz/dt_true = J_encoder(x) @ f(x)
    dz_dt_true_cols = []
    for k in range(z.shape[1]):  # nz
        grad_zk = torch.autograd.grad(
            outputs=z[:, k].sum(),      # scalar
            inputs=Xcol,
            create_graph=True,
            retain_graph=True
        )[0]                            # (B, nx)
        dz_dt_true_k = (grad_zk * fcol).sum(dim=1)  # (B,)
        dz_dt_true_cols.append(dz_dt_true_k)

    dz_dt_true = torch.stack(dz_dt_true_cols, dim=1)  # (B, nz)

    return torch.mean((dz_dt_pred - dz_dt_true) ** 2)



def generatetrajectories(ntraj, tsteps, A, trainflag):

    nx, nz = A.shape
    nt = len(tsteps)

    if trainflag:
        z1 = np.random.uniform(low=-1.5, high=0.5, size=ntraj)
        z2 = np.random.uniform(low=-1, high=1, size=ntraj)
    else:
        z1 = np.random.uniform(low=-1.5, high=1.5, size=ntraj)
        z2 = np.random.uniform(low=-1, high=1, size=ntraj)
    Z0 = np.column_stack((z1, z2))  # ntraj x nz

    Z = np.zeros((ntraj, nt, nz))

    def zode(t, z):
        return [z[1], z[0]-z[0]**3]

    for i in range(ntraj):
        sol = solve_ivp(zode, (tsteps[0], tsteps[-1]), Z0[i, :], t_eval=tsteps)
        Z[i, :, :] = sol.y.T

    # map to high dimensional space
    X = np.zeros((ntraj, nt, nx))
    for i in range(nt):
        X[:, i, :] = Z[:, i, :]**3 @ A.T

    return X


def compute_test_error(model, Xtest, t_test):
    """Compute MSE on test set"""
    Xtest_t = torch.tensor(Xtest, dtype=torch.float32)
    t_test_t = torch.tensor(t_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        X0_test = Xtest_t[:, 0, :]
        Xhat = model.predict_trajectory(X0_test, t_test_t)
    return torch.mean((Xtest_t - Xhat) ** 2).item()


def getdata(ntrain, ntest, ncol, t_train, t_test):

    nz = 2
    nx = 128

    A = np.random.normal(size=(nx, nz))

    Xtrain = generatetrajectories(ntrain, t_train, A, trainflag=True)
    Xtest = generatetrajectories(ntest, t_test, A, trainflag=False)

    # collocation points
    z1 = np.random.uniform(low=0.5, high=1.5, size=ncol)
    z2 = np.random.uniform(low=-1, high=1, size=ncol)
    Zcol = np.column_stack((z1, z2))  # ncol x nz
    hZ = np.column_stack((Zcol[:, 1], Zcol[:, 0] - Zcol[:, 0]**3))
    fcol = np.zeros((ncol, nx))
    for i in range(ncol):
        fcol[i, :] =  hZ[[i], :] @ (3 * A * Zcol[i, :].T**2).T
    Xcol = Zcol**3 @ A.T

    return Xtrain, Xtest, Xcol, fcol, A


def true_encoder(X, A):  # X is npts * nt * nx
    Z3 = X @ np.linalg.pinv(A).T  # pinv is nz x nx
    return np.sign(Z3) * np.abs(Z3)**(1/3)


if __name__ == "__main__":

    # discretization in time for training and test data.  These don't need to be changed.
    nt_train = 11
    nt_test = 21
    t_train = np.linspace(0.0, 1.0, nt_train)
    t_test = np.linspace(0.0, 1.0, nt_test)

    # number of training pts, testing pts, and collocation pts.
    # You will need more training pts and collocation pts eventually (testing pts can remain as is).
    ntrain = 1000
    ntest = 100
    ncol = 1000
    Xtrain, Xtest, Xcol, fcol, Amap = getdata(ntrain, ntest, ncol, t_train, t_test)

    # Xtrain is ntrain x nt_train x nx
    # Xtest is ntest x nt_test x nx
    # Xcol is ncol x nx
    # fcol is ncol x nx and represents f(Xcol)
    # Amap is only needed for final plot (see function below)

    nx = 128  # High-dimensional space
    nz = 2    # Latent space dimension (from paper)

    model = PINODE(nx, nz, width=128, depth=3)

    losses = train(model, Xtrain, t_train, Xcol, fcol, num_epochs=1500, lr=1e-3, Xtest=Xtest, t_test=t_test)

    t_test_t = torch.tensor(t_test, dtype=torch.float32)
    Xtest_t = torch.tensor(Xtest, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        X0_test = Xtest_t[:, 0, :]  # Initial conditions
        Xhat = model.predict_trajectory(X0_test, t_test_t).numpy()

    print(compute_test_error(model, Xtest, t_test))

    # Project predictions back to true latent space for visualization
    Zhat = true_encoder(Xhat, Amap)

    plt.figure()
    for i in range(0, ntest):
        plt.plot(Zhat[i, 0, 0], Zhat[i, 0, 1], "ko")
        plt.plot(Zhat[i, :, 0], Zhat[i, :, 1], "k")
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1, 1])
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Predicted Trajectories in True Latent Space")
    plt.show()
    plt.savefig("pinode_true_latent.png")

    plt.cla()
    plt.plot([l["total"] for l in losses], label="Total Loss")
    plt.plot([l["reconstruction"] for l in losses], label="Reconstruction Loss")
    plt.plot([l["dynamics"] for l in losses], label="Dynamics Loss")
    plt.plot([l["physics"] for l in losses], label="Physics Loss")
    plt.plot([l["test"] for l in losses], label="Test MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.show()
    plt.savefig("pinode_training_losses.png")