import numpy as np
import torch as t
from torch import nn
import matplotlib.pyplot as plt


# ---------------- Model ----------------
class PINN(nn.Module):
    def __init__(self, hlayers, width):
        super(PINN, self).__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, model_input):
        return self.linear_stack(model_input)


# ---------------- Physics ----------------
def physics_residual(model, time, x, params):
    # Burgers: u_t + lambda1*u*u_x - lambda2*u_xx = 0

    lambda1, lambda2 = params

    # Ensure shapes [N] and require grads
    if time.dim() != 1:
        time = time.reshape(-1)
    if x.dim() != 1:
        x = x.reshape(-1)

    # time, x must require grad for autograd to compute derivatives wrt them
    if not time.requires_grad:
        time = time.requires_grad_(True)
    if not x.requires_grad:
        x = x.requires_grad_(True)

    model_input = t.stack((time, x), dim=1)  # [N,2]
    u = model(model_input)                  # [N,1]

    u_t = t.autograd.grad(u, time, grad_outputs=t.ones_like(u), create_graph=True)[0]  # [N]
    u_x = t.autograd.grad(u, x, grad_outputs=t.ones_like(u), create_graph=True)[0]     # [N]
    u_xx = t.autograd.grad(u_x, x, grad_outputs=t.ones_like(u_x), create_graph=True)[0]  # [N]

    # u is [N,1], u_x is [N], so squeeze u
    u_s = u.squeeze(1)  # [N]

    f = u_t + lambda1 * u_s * u_x - lambda2 * u_xx
    return f


# ---------------- Data residuals (IC/BC) ----------------
def initial_condition_residual(model, x0):
    # IC: u(0,x) = -sin(pi x), x in [-1,1]
    t0 = t.zeros_like(x0)
    model_input = t.stack((t0, x0), dim=1)
    u_hat = model(model_input).squeeze(1)
    u_true = -t.sin(t.pi * x0)
    return u_hat - u_true


def boundary_condition_residual(model, tb):
    # BC: u(t,-1)=0 and u(t,1)=0, t in [0,1]
    x_left = -t.ones_like(tb)
    x_right = t.ones_like(tb)

    inp_left = t.stack((tb, x_left), dim=1)
    inp_right = t.stack((tb, x_right), dim=1)

    u_left = model(inp_left).squeeze(1)
    u_right = model(inp_right).squeeze(1)

    # both should be zero
    return u_left, u_right


# ---------------- Point sampling (matches paper structure) ----------------
def sample_ic_points(N0, device=None):
    # x0 ~ Uniform[-1,1]
    x0 = (2.0 * t.rand(N0, device=device) - 1.0)
    return x0


def sample_bc_points(Nb, device=None):
    # tb ~ Uniform[0,1]
    tb = t.rand(Nb, device=device)
    return tb


def sample_collocation_points(Nf, device=None):
    # (t_f, x_f) ~ Uniform([0,1] x [-1,1])
    tf = t.rand(Nf, device=device).requires_grad_(True)
    xf = (2.0 * t.rand(Nf, device=device) - 1.0).requires_grad_(True)
    return tf, xf


# ---------------- Training ----------------
def train_adam(model, params, epochs, lr=1e-3, N0=50, Nb=50, Nf=10000, device=None):
    model.train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Separate datasets like the paper
        x0 = sample_ic_points(N0, device=device)
        tb = sample_bc_points(Nb, device=device)
        tf, xf = sample_collocation_points(Nf, device=device)

        # Residuals
        f = physics_residual(model, tf, xf, params)
        ic_res = initial_condition_residual(model, x0)
        bc_left, bc_right = boundary_condition_residual(model, tb)

        # MSEs (paper-style)
        loss_f = t.mean(f**2)
        loss_ic = t.mean(ic_res**2)
        loss_bc = t.mean(bc_left**2) + t.mean(bc_right**2)

        loss = loss_f + loss_ic + loss_bc
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 200 == 0:
            print(f"Adam epoch {epoch}: total={loss.item():.3e}  f={loss_f.item():.3e}  ic={loss_ic.item():.3e}  bc={loss_bc.item():.3e}")

    return losses


# ---------------- Plotting (paper-like: field + training points overlay) ----------------
def plot_like_paper(model, ic_x0, bc_tb, N=256, title="Burgers PINN (prediction)"):
    model.eval()

    tt = t.linspace(0.0, 1.0, N)
    xx = t.linspace(-1.0, 1.0, N)
    t_mesh, x_mesh = t.meshgrid(tt, xx, indexing="ij")  # [N,N]

    tx = t.stack([t_mesh.reshape(-1), x_mesh.reshape(-1)], dim=1)
    with t.no_grad():
        u = model(tx).reshape(N, N).cpu().numpy()

    t_np = t_mesh.cpu().numpy()
    x_np = x_mesh.cpu().numpy()

    plt.figure(figsize=(10, 4))
    pcm = plt.pcolormesh(t_np, x_np, u, shading="auto")
    plt.colorbar(pcm, label="u(t,x)")

    # Overlay IC points: (t=0, x=ic_x0)
    plt.scatter(
        t.zeros_like(ic_x0).cpu().numpy(),
        ic_x0.cpu().numpy(),
        s=12, c="k", marker="o", label="IC data"
    )

    # Overlay BC points: (t=bc_tb, x=-1) and (t=bc_tb, x=1)
    plt.scatter(bc_tb.cpu().numpy(), (-t.ones_like(bc_tb)).cpu().numpy(), s=12, c="k", marker="x", label="BC data")
    plt.scatter(bc_tb.cpu().numpy(), (t.ones_like(bc_tb)).cpu().numpy(), s=12, c="k", marker="x")

    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()


def plot_loss(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss (Adam)")
    plt.tight_layout()
    plt.show()


# ---------------- Main ----------------
if __name__ == "__main__":
    device = None  # or: device = "cuda" if you want

    lambda1 = 1.0
    lambda2 = 0.01 / np.pi
    params = (lambda1, lambda2)

    # Paper-like architecture often used for Burgers: ~8-9 layers, ~20 neurons
    model = PINN(hlayers=8, width=20)
    if device is not None:
        model = model.to(device)

    # For plotting overlays, keep a copy of the IC/BC point sets you trained on
    N0, Nb, Nf = 50, 50, 10000
    ic_x0 = sample_ic_points(N0, device=device)
    bc_tb = sample_bc_points(Nb, device=device)

    # Adam warm start
    adam_losses = train_adam(
        model, params,
        epochs=1000,
        lr=5e-3,
        N0=N0, Nb=Nb, Nf=Nf,
        device=device
    )

    plot_loss(adam_losses)
    plot_like_paper(model, ic_x0, bc_tb, N=256, title="Burgers equation: u(t,x) with IC/BC training points")
