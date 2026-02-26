# (1+λ)-ES hill-climber with elitism for a Scrabble-like environment.
# - Keep a single champion model θ
# - Each generation: sample λ mutated candidates around θ
# - Evaluate each candidate over K games (multiple seeds)
# - Promote the best candidate if it beats the champion

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Bernoulli
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# -----------------------------
# 1) Policy network (stochastic multi-head)
# -----------------------------
class StochasticScrabblePolicy(nn.Module):
    """
    Outputs:
      - 7 categorical heads x 26 options
      - 2 categorical heads x 14 options
      - 1 Bernoulli button
    """

    def __init__(self, obs_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        d = obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.Tanh()]
            d = hidden
        self.trunk = nn.Sequential(*layers)

        self.head_26 = nn.Linear(hidden, 7 * 26)
        self.head_14 = nn.Linear(hidden, 2 * 14)
        self.head_btn = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.trunk(obs)
        logits_26 = self.head_26(x).view(-1, 7, 26)
        logits_14 = self.head_14(x).view(-1, 2, 14)
        logit_btn = self.head_btn(x).view(-1, 1)
        return logits_26, logits_14, logit_btn

    @torch.no_grad()
    def act(
            self,
            obs,
            temperature: float = 1.0,
            stochastic: bool = True,
            masks_26: Optional[torch.Tensor] = None,  # (7,26) or (B,7,26)
            masks_14: Optional[torch.Tensor] = None,  # (2,14) or (B,2,14)
            device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if o.dim() == 1:
            o = o.unsqueeze(0)  # (1, obs_dim)

        logits_26, logits_14, logit_btn = self.forward(o)

        temp = max(float(temperature), 1e-6)
        logits_26 = logits_26 / temp
        logits_14 = logits_14 / temp
        logit_btn = logit_btn / temp

        # Optional masks: True = valid
        if masks_26 is not None:
            m26 = masks_26.to(device)
            if m26.dim() == 2:
                m26 = m26.unsqueeze(0)
            logits_26 = logits_26.masked_fill(~m26, float("-inf"))

        if masks_14 is not None:
            m14 = masks_14.to(device)
            if m14.dim() == 2:
                m14 = m14.unsqueeze(0)
            logits_14 = logits_14.masked_fill(~m14, float("-inf"))

        logits_26 = logits_26.squeeze(0)  # (7,26)
        logits_14 = logits_14.squeeze(0)  # (2,14)
        logit_btn = logit_btn.squeeze(0)  # (1,)

        if stochastic:
            a26 = Categorical(logits=logits_26).sample()
            a14 = Categorical(logits=logits_14).sample()
            btn = Bernoulli(logits=logit_btn).sample().to(torch.int64)
        else:
            a26 = logits_26.argmax(dim=-1)
            a14 = logits_14.argmax(dim=-1)
            btn = (torch.sigmoid(logit_btn) > 0.5).to(torch.int64)

        return {"a26": a26.cpu().tolist(), "a14": a14.cpu().tolist(), "btn": int(btn.item())}


# -----------------------------
# 2) Parameter vector helpers
# -----------------------------
def get_theta(model: nn.Module, device: torch.device | str = "cpu") -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().to(device).clone()


def set_theta(model: nn.Module, theta: torch.Tensor) -> None:
    vector_to_parameters(theta, model.parameters())


def theta_noise_like(theta: torch.Tensor, sigma: float) -> torch.Tensor:
    return sigma * torch.randn_like(theta)


# -----------------------------
# 3) Reward aggregation
# -----------------------------
def episodic_reward(total_score: float, win: bool, win_weight: float) -> float:
    # score transform keeps scale sane; change if you know score range
    return math.log1p(max(0.0, float(total_score))) + win_weight * (1.0 if win else 0.0)


# -----------------------------
# 4) Environment contract you must implement
# -----------------------------
class ScrabbleEnv:
    """
    You implement this interface.

    Required:
      reset(seed) -> obs
      step(action_dict) -> obs, turn_score, done, info

    info should include:
      info["win"] at terminal step (bool)   OR you can set it via env methods
    """

    def reset(self, seed: int | None = None):
        pass


    def step(self, action: Dict[str, Any]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError


# -----------------------------
# 5) Evaluate a policy (K games)
# -----------------------------
@torch.no_grad()
def eval_theta(
        env_factory: Callable[[], ScrabbleEnv],
        model_ctor: Callable[[], StochasticScrabblePolicy],
        theta: torch.Tensor,
        seeds: List[int],
        device: torch.device | str = "cpu",
        temperature: float = 1.0,
        stochastic_actions: bool = True,
        win_weight: float = 3.0,
        max_steps: int = 10_000,
) -> float:
    model = model_ctor().to(device)
    set_theta(model, theta)

    returns: List[float] = []
    for sd in seeds:
        env = env_factory()
        obs = env.reset(seed=sd)

        total_score = 0.0
        done = False
        steps = 0
        win = False

        while not done and steps < max_steps:
            action = model.act(obs, temperature=temperature, stochastic=stochastic_actions, device=device)
            obs, turn_score, done, info = env.step(action)
            total_score += float(turn_score)
            if done:
                win = bool(info.get("win", False))
            steps += 1

        returns.append(episodic_reward(total_score, win, win_weight))

    return float(np.mean(returns))


# -----------------------------
# 6) (1+λ)-ES hill climber with elitism
# -----------------------------
@dataclass
class TrainConfig:
    generations: int = 500
    lam: int = 64  # candidates per generation
    sigma: float = 0.02  # mutation std on theta
    k_games: int = 5  # rollouts per candidate
    win_weight: float = 3.0
    temperature_train: float = 1.2
    temperature_eval: float = 0.5
    promote_margin: float = 0.0  # require candidate to exceed champ by this margin to replace
    device: str = "cpu"
    seed0: int = 1234


def train_es_hillclimb(
        env_factory: Callable[[], ScrabbleEnv],
        model_ctor: Callable[[], StochasticScrabblePolicy],
        cfg: TrainConfig,
) -> Tuple[StochasticScrabblePolicy, Dict[str, Any]]:
    device = torch.device(cfg.device)

    # Initialize champion
    champ_model = model_ctor().to(device)
    champ_theta = get_theta(champ_model, device=device)

    rng = random.Random(cfg.seed0)

    # Baseline eval seeds used to compare within a generation (controls variance)
    def make_seeds(base: int, n: int) -> List[int]:
        return [base + i for i in range(n)]

    # Evaluate initial champion
    base_seed = cfg.seed0 * 100_000
    champ_fit = eval_theta(
        env_factory, model_ctor, champ_theta,
        seeds=make_seeds(base_seed, cfg.k_games),
        device=device,
        temperature=cfg.temperature_eval,
        stochastic_actions=False,
        win_weight=cfg.win_weight,
    )

    best_fit_ever = champ_fit
    best_theta_ever = champ_theta.clone()

    for gen in range(cfg.generations):
        # Use consistent seeds across champ + candidates for fair comparisons this gen
        gen_seed_base = base_seed + (gen + 1) * 10_000
        seeds = make_seeds(gen_seed_base, cfg.k_games)

        # Re-evaluate champ on these seeds (keeps comparisons fair)
        champ_fit_gen = eval_theta(
            env_factory, model_ctor, champ_theta, seeds,
            device=device,
            temperature=cfg.temperature_eval,
            stochastic_actions=False,
            win_weight=cfg.win_weight,
        )

        # Sample λ candidates around champ
        cand_thetas: List[torch.Tensor] = []
        for _ in range(cfg.lam):
            noise = theta_noise_like(champ_theta, cfg.sigma)
            cand_thetas.append(champ_theta + noise)

        # Evaluate candidates (stochastic actions for exploration)
        cand_fits = []
        for i, th in enumerate(cand_thetas):
            fit = eval_theta(
                env_factory, model_ctor, th, seeds,
                device=device,
                temperature=cfg.temperature_train,
                stochastic_actions=True,
                win_weight=cfg.win_weight,
            )
            cand_fits.append(fit)

        # Choose best candidate
        best_idx = int(np.argmax(cand_fits))
        best_cand_fit = float(cand_fits[best_idx])
        best_cand_theta = cand_thetas[best_idx]

        # Promotion decision
        improved = best_cand_fit > (champ_fit_gen + cfg.promote_margin)
        if improved:
            champ_theta = best_cand_theta.detach().clone()
            champ_fit = best_cand_fit

        # Track best-ever using deterministic eval signal (champ_fit_gen) OR best_cand_fit (training signal).
        if champ_fit_gen > best_fit_ever:
            best_fit_ever = champ_fit_gen
            best_theta_ever = champ_theta.clone()

        print(
            f"gen {gen:04d} | champ_eval {champ_fit_gen:8.3f} | "
            f"best_cand {best_cand_fit:8.3f} | promoted {improved}"
        )

    # Return best-ever champion as a model
    final_model = model_ctor().to(device)
    set_theta(final_model, best_theta_ever)

    stats = {
        "best_fitness_eval": best_fit_ever,
        "final_champ_fitness_training": champ_fit,
        "theta_dim": int(best_theta_ever.numel()),
    }
    return final_model, stats


# -----------------------------
# 7) How you hook this up
# -----------------------------
# You provide:
#   - env_factory(): creates a new ScrabbleEnv instance
#   - model_ctor(): constructs policy with the correct obs_dim


if __name__ == "__main__":
    # Example placeholders; you must implement ScrabbleEnv and obs_dim.
    OBS_DIM = 15 * 15 + 7 + 2 + 1  # this is 1 for every tile on the board, the tiles in their hand, their score,
    # the opponent score, and finally the tiles left in the bag

    def model_ctor() -> StochasticScrabblePolicy:
        return StochasticScrabblePolicy(obs_dim=OBS_DIM, hidden=256, depth=3)


    def env_factory() -> ScrabbleEnv:
        # return YourScrabbleEnv(...)
        raise NotImplementedError("Implement env_factory() to return your ScrabbleEnv")


    cfg = TrainConfig(
        generations=200,
        lam=64,
        sigma=0.02,
        k_games=5,
        win_weight=3.0,
        temperature_train=1.2,
        temperature_eval=0.5,
        promote_margin=0.0,
        device="cpu",
        seed0=1234,
    )
