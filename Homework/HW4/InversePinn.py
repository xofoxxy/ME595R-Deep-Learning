import numpy as np
import torch as t
from torch import nn
import matplotlib.pyplot as plt


class PINN(nn.Module):
    def __init__(self, hlayers, width):
        super(PINN, self).__init__()
        self.lambda1 = nn.Parameter(t.tensor(1.0))
        self.lambda2 = nn.Parameter(t.tensor(-1.0))  # lambda2 is
        self.flatten = nn.Flatten()
        layers = [nn.Linear(2, width), nn.Tanh()]

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.linear_stack = nn.Sequential(*layers)

    def forward(self, model_input):  # model_input allows a single value (tensor 2, n_col)
        return self.linear_stack(model_input)


# ---------------- Physics ----------------
def physics_residual(model, time, x, params):

    """Burgers: u_t + lambda1*u*u_x - lambda2*u_xx = 0"""
    # Ensure time and x are 1D and require grad
    time = time.reshape(-1).requires_grad_(True)
    x = x.reshape(-1).requires_grad_(True)

    model_input = t.stack((time, x), dim=1)
    u = model(model_input)[:, 0]

    l1 = model.lambda1
    l2 = model.lambda2

    u_t = t.autograd.grad(u, time, grad_outputs=t.ones_like(u), create_graph=True)[0]  # [N]
    u_x = t.autograd.grad(u, x, grad_outputs=t.ones_like(u), create_graph=True)[0]  # [N]
    u_xx = t.autograd.grad(u_x, x, grad_outputs=t.ones_like(u_x), create_graph=True)[0]  # [N]

    f = u_t + l1 * u * u_x - l2 * u_xx
    return f


def initial_condition_residual(model, x):
    t_0 = t.zeros_like(x)  # t=0
    model_input = t.stack((t_0, x), dim=1)
    u_pred = model(model_input)[:, 0]

    # True IC: u(0,x) = -sin(πx)
    u_true = -t.sin(np.pi * x)

    return u_pred - u_true


def boundary_condition_residual(model, tb):
    # BC: u(t,-1)=0 and u(t,1)=0, t in [0,1]
    x_left = -t.ones_like(tb)
    x_right = t.ones_like(tb)

    inp_left = t.stack((tb, x_left), dim=1)
    inp_right = t.stack((tb, x_right), dim=1)

    u_left = model(inp_left)[:, 0]
    u_right = model(inp_right)[:, 0]

    # both should be zero
    return u_left, u_right


#
# def physics_residual(model, time, x, params):  # t: ncol * 1, x: ncol * 1
#     # Burger's equation: u_t + uu_x − l2*u_x2 =0
#
#     l1, l2 = params
#
#     model_input = t.stack((time, x), dim=1)
#     # print("Model Input Shape: " + str(model_input.shape))
#
#     y = model(model_input)
#
#     dydx = t.autograd.grad(y, x, grad_outputs=t.ones_like(y), create_graph=True)[0]
#     d2ydx = t.autograd.grad(dydx, x, grad_outputs=t.ones_like(dydx), create_graph=True)[0]
#
#     dydt = t.autograd.grad(y, time, grad_outputs=t.ones_like(y), create_graph=True)[0]
#
#     return dydt + l1*y*dydx - l2*d2ydx
#
#
# def boundary_residual(model, time, x, params):
#
#     # we don't ever use these but for consistency we're going to unpack them
#     l1, l2 = params
#
#     # Boundary conditions: u(t=-1, x) = u(t=1, x) = 0
#
#     x_left = t.ones_like(time) * -1
#     x_right = t.ones_like(time)
#
#     bc_inputs_left = t.stack((time, x_left), dim=1)
#     bc_inputs_right = t.stack((time, x_right), dim=1)
#
#     y_left = model(bc_inputs_left)
#     y_right = model(bc_inputs_right)
#
#     # initial conditions: u(t=0, x) = 1
#
#     t_0 = t.zeros_like(x)
#     bc_inputs_0 = t.stack((t_0, x), dim=1)
#     y_0_hat = model(bc_inputs_0)
#
#     y_0 = -1 * t.sin(t.pi * x)
#
#     # In theory these are what we want to optimize
#     return y_left, y_right, y_0 - y_0_hat


def generate_collocation_points(n_points=500):
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=2, seed=69420)  # lol nice
    samples = sampler.random(n_points)

    samples = t.tensor(samples, dtype=t.float32)

    # Scale samples to the desired ranges
    t_samples = samples[:, 0] * 1.0  # Scale to [0, 1]
    x_samples = samples[:, 1] * 2.0 - 1.0  # Scale to [-1, 1]

    t_flat = t_samples.requires_grad_(True)
    x_flat = x_samples.requires_grad_(True)

    # then we flatten them into 1d tensors and then we stack them together for our model input
    collocation_points = t.stack((t_flat, x_flat), dim=1)

    return collocation_points


def load_measurements():
    data = np.loadtxt("data.txt")

    # Columns: x, t, u
    x_data = t.tensor(data[:, 0], dtype=t.float32).reshape(-1, 1)
    t_data = t.tensor(data[:, 1], dtype=t.float32).reshape(-1, 1)
    u_data = t.tensor(data[:, 2], dtype=t.float32).reshape(-1, 1)

    x_data.requires_grad = True
    t_data.requires_grad = True

    return t_data, x_data, u_data


def data_residual(model, t_data, x_data, u_data):
    model_input = t.stack((t_data.reshape(-1), x_data.reshape(-1)), dim=1)
    u_pred = model(model_input)[:, 0]

    return u_pred - u_data.reshape(-1)


def train(model, optimizer, epochs, params):
    alpha1 = 5
    alpha2 = 5  # alpha1 is for the boundary loss, alpha2 is for the initial condition loss
    alpha3 = 2.5  # alpha3 is for the data loss

    losses = t.zeros(epochs, 3)  # [phys_loss, bc_loss, ic_loss, total_loss]
    lambdas = t.zeros(epochs, 2)

    # Generate training points and collocation points
    c_points = generate_collocation_points()

    t_data, x_data, u_data = load_measurements()

    for epoch in range(epochs):
        optimizer.zero_grad()

        t_flat = c_points[:, 0]  # slice the first dimension
        x_flat = c_points[:, 1]  # slice the second dimension

        # Calculate residuals
        physics_res = physics_residual(model, t_flat, x_flat, params)

        data_res = data_residual(model, t_data, x_data, u_data)

        # Compute losses
        phys_loss = t.mean(physics_res ** 2)
        data_loss = t.mean(data_res ** 2)

        sum_of_loss = phys_loss + alpha3 * data_loss

        losses[epoch, 0] = phys_loss.item()
        losses[epoch, 1] = data_loss.item()
        # losses[epoch, 2] = ic_loss.item()
        losses[epoch, 2] = sum_of_loss.item()

        sum_of_loss.backward()

        lambdas[epoch, 0] = model.lambda1.item()
        lambdas[epoch, 1] = model.lambda2.item()

        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Total={sum_of_loss.item():.4f}, Data={data_loss.item():.4f}, "
                  f"Physics={phys_loss.item():.4f}, λ₁={model.lambda1.item():.4f}, λ₂={model.lambda2.item():.6f}")

    return losses, lambdas


def n_plot(model, xmin=-1, xmax=1, time=0, N=100):
    plt.cla()
    plt.xlim(xmin, xmax)
    plt.ylim(-1, 1)  # I love magic numbers

    plt.figure(figsize=(10, 10))

    x_range = t.linspace(xmin, xmax, N)
    t_range = t.zeros_like(x_range)

    model_input = t.stack((t_range, x_range), dim=1)

    u_hat = model(model_input)[:, 0]

    plt.plot(x_range, u_hat.tolist())
    plt.xlabel("x")
    plt.ylabel("u_hat")

    plt.annotate(f"T = {time}", (.6, .6))
    plt.savefig("N_plot.png")


def plot_like_paper(
        model,
        N=256,
        x_min=-1.0,
        x_max=1.0,
        t_min=0.0,
        t_max=1.0,
        ic_points=None,  # tuple (t0, x0) or None
        bc_points=None,  # tuple (tb, xb) or None
        title="Burgers' equation (PINN prediction)"
):
    """
    Once again Chat did this for me. but the model was all mine.

    Produces a paper-style spatio-temporal plot of u(t,x) over the domain,
    with optional overlays of initial-condition and boundary-condition training points.

    ic_points: (t0, x0) where both are 1D torch tensors of same length
               (typically t0 all zeros).
    bc_points: (tb, xb) where both are 1D torch tensors of same length
               (typically xb is -1 or +1).
    """

    model.eval()

    # Dense evaluation grid
    tt = t.linspace(t_min, t_max, N)
    xx = t.linspace(x_min, x_max, N)
    t_mesh, x_mesh = t.meshgrid(tt, xx, indexing="ij")  # shape [N, N]

    tx = t.stack([t_mesh.reshape(-1), x_mesh.reshape(-1)], dim=1)  # [N*N, 2]

    with t.no_grad():
        u = model(tx)[:, 0].reshape(N, N).cpu().numpy()

    # Convert grids to numpy for plotting
    t_np = t_mesh.cpu().numpy()
    x_np = x_mesh.cpu().numpy()

    plt.figure(figsize=(10, 4))

    # Use a heatmap-like view (similar to paper figures)
    # pcolormesh lets us use (t,x) coordinates directly
    pcm = plt.pcolormesh(t_np, x_np, u, shading="auto")
    plt.colorbar(pcm, label="u(t, x)")

    # Overlay training points (like the paper)
    if ic_points is not None:
        t0, x0 = ic_points
        plt.scatter(
            t0.detach().cpu().numpy(),
            x0.detach().cpu().numpy(),
            s=12,
            c="k",
            marker="o",
            label="IC data"
        )

    if bc_points is not None:
        tb, xb = bc_points
        plt.scatter(
            tb.detach().cpu().numpy(),
            xb.detach().cpu().numpy(),
            s=12,
            c="k",
            marker="x",
            label="BC data"
        )

    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)

    # Only show legend if we actually added point overlays
    if ic_points is not None or bc_points is not None:
        plt.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig("pressurePlot.jpeg", dpi=600)

    # Optional: also show a few time-slice line plots (often included under the heatmap)
    # Uncomment if you want that second panel style.
    """
    plt.figure(figsize=(8, 4))
    for t_snap in [0.25, 0.50, 0.75]:
        t_line = t.ones(N) * t_snap
        x_line = t.linspace(x_min, x_max, N)
        tx_line = t.stack([t_line, x_line], dim=1)
        with t.no_grad():
            u_line = model(tx_line).squeeze().cpu().numpy()
        plt.plot(x_line.cpu().numpy(), u_line, label=f"t={t_snap:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(t,x)")
    plt.title("Predicted solution slices")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """


def losses_plot(losses, lambdas):
    plt.figure(figsize=(12, 5))

    # Plot 1: Losses
    plt.subplot(1, 2, 1)
    plt.plot(losses[:, 2], label="total loss")
    plt.plot(losses[:, 0], label="physics loss")
    plt.plot(losses[:, 1], label="data loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Loss Components")

    # Plot 2: Lambda Evolution
    plt.subplot(1, 2, 2)
    plt.plot(lambdas[:, 0], label="lambda1 (learned)")
    plt.plot(lambdas[:, 1], label="lambda2 (learned)")
    plt.axhline(y=1.0, color='r', linestyle='--', label="lambda1 (true)")
    plt.axhline(y=-0.01 / np.pi, color='b', linestyle='--', label="lambda2 (true)")
    plt.xlabel("epoch")
    plt.ylabel("lambda value")
    plt.legend()
    plt.title("Parameter Discovery")

    plt.tight_layout()
    plt.savefig("training_diagnostics.jpeg", dpi=300)

if __name__ == "__main__":
    lambda1 = 1;
    lambda2 = 0.01 / np.pi
    params = (lambda1, lambda2)

    model = PINN(4, 64)

    optimizer = t.optim.Adam(model.parameters(), lr=0.0025)

    epochs = 10000

    losses = train(model, optimizer, epochs, params)

    plot_like_paper(model)

    n_plot(model, time=0.75)

    losses_plot(*losses)
