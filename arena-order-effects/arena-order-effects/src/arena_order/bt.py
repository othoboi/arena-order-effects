import numpy as np
from scipy import stats


def build_design(left, right, models):
    index = {name: i for i, name in enumerate(models)}
    n_votes = len(left)
    n_cols = len(models) - 1  # last model = ref (0), else not identifiable
    X = np.zeros((n_votes, n_cols))
    for row, (l, r) in enumerate(zip(left, right)):
        li, ri = index[l], index[r]
        if li < n_cols:
            X[row, li] += 1.0
        if ri < n_cols:
            X[row, ri] -= 1.0
    return X


def fit_logistic(X, y, max_iter=100, tol=1e-10, ridge=1e-8):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        grad = X.T @ (y - p)
        hess = (X * w[:, None]).T @ X + ridge * np.eye(X.shape[1])  # tiny ridge, numerics only
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1.0 - p)
    cov = np.linalg.inv((X * w[:, None]).T @ X + ridge * np.eye(X.shape[1]))
    return beta, cov, loglik


def fit_side_effect(left, right, left_won, covariates=None):
    left = list(left)
    right = list(right)
    y = np.asarray(left_won, dtype=float)
    models = sorted(set(left) | set(right))

    X_strength = build_design(left, right, models)
    n_strength = X_strength.shape[1]
    names = list(covariates) if covariates else []
    if names:
        extra = np.column_stack([np.asarray(covariates[k], dtype=float) for k in names])
        X_strength = np.hstack([X_strength, extra])

    beta0, _, loglik0 = fit_logistic(X_strength, y)

    X_full = np.hstack([X_strength, np.ones((len(y), 1))])
    beta, cov, loglik1 = fit_logistic(X_full, y)

    gamma = float(beta[-1])
    se = float(np.sqrt(cov[-1, -1]))
    lrt = 2.0 * (loglik1 - loglik0)
    p_value = float(stats.chi2.sf(lrt, df=1))  # LRT, 1 df (just gamma)

    strengths = dict(zip(models[:-1], beta[:n_strength]))
    strengths[models[-1]] = 0.0
    strengths_no_side = dict(zip(models[:-1], beta0[:n_strength]))
    strengths_no_side[models[-1]] = 0.0
    cov_out = {}
    for k, name in enumerate(names):
        j = n_strength + k
        cov_out[name] = {"coef": float(beta[j]), "se": float(np.sqrt(cov[j, j]))}

    return {
        "n_votes": int(len(y)),
        "n_models": len(models),
        "left_win_share": float(y.mean()),
        "gamma": gamma,
        "se": se,
        "ci_low": gamma - 1.96 * se,
        "ci_high": gamma + 1.96 * se,
        "p_left_when_even": float(1.0 / (1.0 + np.exp(-gamma))),
        "lrt": float(lrt),
        "p_value": p_value,
        "strengths": strengths,
        "strengths_no_side": strengths_no_side,
        "covariates": cov_out,
    }


def cluster_bootstrap_gamma(left, right, left_won, cluster, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    left = np.asarray(left)
    right = np.asarray(right)
    y = np.asarray(left_won)
    cluster = np.asarray(cluster)

    ids, inverse = np.unique(cluster, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(len(ids))]

    gammas = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(ids), size=len(ids))  # resample ppl, not votes :)
        rows = np.concatenate([rows_of[k] for k in picked])
        fit = fit_side_effect(left[rows], right[rows], y[rows])
        gammas.append(fit["gamma"])
    low, high = np.percentile(gammas, [2.5, 97.5])
    return float(low), float(high)


def voter_heterogeneity(left_won, voter, min_votes=20):
    y = np.asarray(left_won)
    voter = np.asarray(voter)
    ids, inverse, counts = np.unique(voter, return_inverse=True, return_counts=True)
    left_counts = np.bincount(inverse, weights=y)
    keep = counts >= min_votes
    n = counts[keep]
    k = left_counts[keep]
    if keep.sum() < 2:
        return {"n_voters": int(keep.sum())}
    pooled = k.sum() / n.sum()
    expected_left = n * pooled
    expected_right = n * (1 - pooled)
    chi2 = float(np.sum((k - expected_left) ** 2 / expected_left
                        + ((n - k) - expected_right) ** 2 / expected_right))
    df = int(keep.sum() - 1)
    share = k / n
    return {
        "n_voters": int(keep.sum()),
        "pooled_left_share": float(pooled),
        "chi2": chi2,
        "df": df,
        "p_value": float(stats.chi2.sf(chi2, df)),
        "voters_above_70pct_left": int(np.sum(share > 0.70)),
        "voters_below_30pct_left": int(np.sum(share < 0.30)),
    }
