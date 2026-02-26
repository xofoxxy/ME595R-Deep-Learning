
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable, Tuple
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ConvectionBC:
    h: float
    T_inf: float


@dataclass(frozen=True)
class AdiabaticBC:
    pass


# -------------------------
# Node object
# -------------------------

@dataclass
class Node:
    node_id: int
    x: float
    y: float
    T: float = 0.0

    neighbors: Dict[str, Optional["Node"]] = field(default_factory=dict)
    bcs: Dict[str, Optional[object]] = field(default_factory=dict)

    def _update(self, k: float, dx: float, dy: float) -> float:

        aP = 0.0
        rhs = 0.0

        face_data = {
            "W": (dx, dy),  # (distance to neighbor, face-length)
            "E": (dx, dy),
            "N": (dy, dx),
            "S": (dy, dx),
        }

        for face, (dist, face_len) in face_data.items():
            nb = self.neighbors.get(face)
            bc = self.bcs.get(face)
            A = face_len * 1.0  # unit depth into page

            if nb is not None:
                a = k * A / dist
                aP += a
                rhs += a * nb.T
                continue

            # no neighbor: must be either adiabatic/symmetry, convection, or nothing
            if bc is None or isinstance(bc, AdiabaticBC):
                continue

            if isinstance(bc, ConvectionBC):
                a = bc.h * A
                aP += a
                rhs += a * bc.T_inf
                continue

            raise ValueError(f"Unknown BC type on face {face}: {type(bc)}")
        self.T = rhs / aP
        return self.T

class FDMModel:
    def __init__(self, k: float, dx: float, dy: float):
        self.k = k
        self.dx = dx
        self.dy = dy
        self.nodes: Dict[int, Node] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def iter_nodes_in_order(self):
        for nid in sorted(self.nodes.keys()):
            yield self.nodes[nid]

    def solve(self, tolerance=1e-10, max_iter=50000):
        for it in range(max_iter):
            max_delta = 0.0

            for node in self.iter_nodes_in_order():
                T_old = node.T
                T_new = node._update(self.k, self.dx, self.dy)

                max_delta = max(max_delta, abs(node.T - T_old))

            if max_delta < tolerance:
                return it + 1, max_delta

        return max_iter, max_delta


def turbine_model(*, k: float, ho: float, T_inf_o: float, hi: float, T_inf_i: float, dx: float = 1e-3,dy: float = 1e-3,):

    model = FDMModel(k=k, dx=dx, dy=dy)

    coords = {}

    # y = 0: nodes 1..6, x = 0..5
    nid = 1
    for ix in range(6):
        coords[nid] = (ix * dx, 0 * dy)
        nid += 1

    # y = 1: nodes 7..12
    for ix in range(6):
        coords[nid] = (ix * dx, 1 * dy)
        nid += 1

    # y = 2: nodes 13..18
    for ix in range(6):
        coords[nid] = (ix * dx, 2 * dy)
        nid += 1

    # y = 3: nodes 19..21 (x = 0..2 only)
    for ix in range(3):
        coords[nid] = (ix * dx, 3 * dy)
        nid += 1

    # create nodes with a reasonable initial guess
    T0 = 0.5 * (T_inf_o + T_inf_i)
    for node_id, (x, y) in coords.items():
        model.add_node(Node(node_id=node_id, x=x, y=y, T=T0))

    # convenience lookup by integer grid indices
    index = {}
    for node_id, (x, y) in coords.items():
        ix = int(round(x / dx))
        iy = int(round(y / dy))
        index[(ix, iy)] = node_id

    # connect neighbors (W/E/N/S) where nodes exist
    offsets = {"W": (-1, 0), "E": (1, 0), "N": (0, -1), "S": (0, 1)}
    for (ix, iy), node_id in index.items():
        node = model.nodes[node_id]
        for face, (ox, oy) in offsets.items():
            nb_id = index.get((ix + ox, iy + oy))
            node.neighbors[face] = model.nodes[nb_id] if nb_id is not None else None

    # -------------------------
    # Apply boundary conditions
    # -------------------------

    for ix in range(6):
        node_id = index[(ix, 0)]
        model.nodes[node_id].bcs["N"] = ConvectionBC(h=ho, T_inf=T_inf_o)

    for iy in range(4):
        if (0, iy) in index:
            model.nodes[index[(0, iy)]].bcs["W"] = AdiabaticBC()

    for iy in range(3):
        model.nodes[index[(5, iy)]].bcs["E"] = AdiabaticBC()

    for ix in range(3):
        model.nodes[index[(ix, 3)]].bcs["S"] = AdiabaticBC()

    for ix in range(2, 6):
        node_id = index[(ix, 2)]
        model.nodes[node_id].bcs["S"] = ConvectionBC(h=hi, T_inf=T_inf_i)

    for iy in (2, 3):
        if (2, iy) in index:
            node_id = index[(2, iy)]
            model.nodes[node_id].bcs["E"] = ConvectionBC(h=hi, T_inf=T_inf_i)

    return model


def channel_heat_transfer_per_length(model: FDMModel, hi: float, T_inf_i: float) -> float:
    q_quarter = 0.0

    for node in model.nodes.values():
        bcS = node.bcs.get("S")
        if isinstance(bcS, ConvectionBC) and abs(node.y - 2e-3) < 1e-12:
            if abs(bcS.h - hi) < 1e-12 and abs(bcS.T_inf - T_inf_i) < 1e-12:
                A = model.dx * 1.0
                q_quarter += hi * A * (node.T - T_inf_i)

        # east face convection on x=2 line => channel left wall
        bcE = node.bcs.get("E")
        if isinstance(bcE, ConvectionBC) and abs(node.x - 2e-3) < 1e-12:
            if abs(bcE.h - hi) < 1e-12 and abs(bcE.T_inf - T_inf_i) < 1e-12:
                A = model.dy * 1.0
                q_quarter += hi * A * (node.T - T_inf_i)

    return 4.0 * q_quarter


def plot_model(node_model):
    points = []

    # Calculate array dimensions based on actual grid spacing
    nx = int(round(5 * node_model.dx / node_model.dx)) + 1  # 6 columns
    ny = int(round(3 * node_model.dy / node_model.dy)) + 1  # 4 rows
    temps = np.full((ny, nx), np.nan)

    for node in node_model.nodes.values():
        x, y = node.x, node.y
        T = node.T
        ix = int(round(x / node_model.dx))
        iy = int(round(y / node_model.dy))
        temps[iy][ix] = T

        points.append((x, y))

    plt.cla()
    plt.scatter(*zip(*points), c=[n.T for n in node_model.nodes.values()])
    # Draw mesh edges
    for node in node_model.nodes.values():
        for face, neighbor in node.neighbors.items():
            if neighbor is not None:
                # Only draw each edge once (check if we haven't drawn it already)
                if face in ["E", "S"]:  # Only draw right and down edges
                    plt.plot([node.x, neighbor.x], [node.y, neighbor.y], 'k-', linewidth=0.5, alpha=0.3)
    plt.colorbar(label="Temperature (K)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("position of nodes and their temperatures")

    plt.show()

    plt.cla()
    plt.imshow(temps, origin='lower')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Temperature distribution")
    plt.yticks(np.arange(ny), [f"{i*node_model.dy}" for i in range(ny)])
    plt.xticks(np.arange(nx), [f"{i*node_model.dx}" for i in range(nx)])
    plt.colorbar(label="Temperature (K)")
    plt.show()


if __name__ == "__main__":

    T_inf_o = 1700.0
    T_inf_i = 400.0
    ho = 788.0

    cases = [
        ("k=11.4, hi=200 (part a)", 11.4, 200.0),
        ("k=11.4, hi=750",           11.4, 750.0),
        ("k=50,   hi=200",           50.0, 200.0),
        ("k=50,   hi=750",           50.0, 750.0),
    ]

    for name, k, hi in cases:
        m = turbine_model(k=k, ho=ho, T_inf_o=T_inf_o, hi=hi, T_inf_i=T_inf_i, dx=1e-3, dy=1e-3)
        iters, err = m.solve(tolerance=1e-10, max_iter=50000)

        # max temperature + node id
        Tmax_node = max(m.nodes.values(), key=lambda n: n.T)
        Tmax = Tmax_node.T

        qprime = channel_heat_transfer_per_length(m, hi=hi, T_inf_i=T_inf_i)

        print(f"{name:22s} | iters={iters:5d} | Tmax={Tmax:8.2f} K @ node {Tmax_node.node_id:2d} {Tmax_node.x, Tmax_node.y} | q'={qprime:10.2f} W/m")
        plot_model(m)
        input()