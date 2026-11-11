import numpy as np


def simulate_votes(n_votes=20000, n_models=12, gamma=0.0, n_voters=500, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"model_{i:02d}" for i in range(n_models)]
    strength = rng.normal(0.0, 0.7, size=n_models)
    left_idx = rng.integers(0, n_models, size=n_votes)
    right_idx = rng.integers(0, n_models - 1, size=n_votes)
    right_idx = np.where(right_idx >= left_idx, right_idx + 1, right_idx)
    eta = strength[left_idx] - strength[right_idx] + gamma
    left_won = (rng.random(n_votes) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    voter = rng.integers(0, n_voters, size=n_votes)
    left = [names[i] for i in left_idx]
    right = [names[i] for i in right_idx]
    return left, right, left_won, voter
