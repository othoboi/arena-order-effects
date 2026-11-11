import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arena_order.bt import fit_side_effect, cluster_bootstrap_gamma
from arena_order.simulate import simulate_votes

LEN_A = "conv_metadata.sum_assistant_a_tokens"
LEN_B = "conv_metadata.sum_assistant_b_tokens"


def balance_check(votes):
    left = votes["model_a"].value_counts()
    right = votes["model_b"].value_counts()
    table = pd.DataFrame({"left": left, "right": right}).fillna(0)
    total = table["left"] + table["right"]
    one_sided = sorted(table.index[(table["left"] == 0) | (table["right"] == 0)])  # never randomized -> drop em
    rest = table.drop(index=one_sided)
    tot = rest["left"] + rest["right"]
    chi2 = float((((rest["left"] - tot / 2) ** 2 + (rest["right"] - tot / 2) ** 2) / (tot / 2)).sum())
    big = rest[tot >= 1000]
    share = big["left"] / (big["left"] + big["right"])
    return one_sided, {
        "one_sided_models": {m: int(total[m]) for m in one_sided},
        "chi2": chi2, "df": int(len(rest)), "p_value": float(stats.chi2.sf(chi2, len(rest))),
        "left_share_range_models_1000plus": [float(share.min()), float(share.max())] if len(big) else None,
    }


def row(label, frame, min_votes=500):
    if len(frame) < min_votes:
        return None
    seen = pd.concat([frame["model_a"], frame["model_b"]]).value_counts()
    ok = seen[seen >= 20].index
    frame = frame[frame["model_a"].isin(ok) & frame["model_b"].isin(ok)]
    fit = fit_side_effect(frame["model_a"], frame["model_b"], frame["left_won"])
    return {"group": label, "n_votes": fit["n_votes"], "left_win_share": fit["left_win_share"],
            "gamma": fit["gamma"], "ci_low": fit["ci_low"], "ci_high": fit["ci_high"],
            "p_left_when_even": fit["p_left_when_even"], "p_value": fit["p_value"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/votes.csv.gz")
    parser.add_argument("--simulate", type=float, default=None,
                        help="skip the real data; simulate votes with this side effect")
    parser.add_argument("--boot", type=int, default=300)
    args = parser.parse_args()
    result = {}

    if args.simulate is not None:
        left, right, y, voter = simulate_votes(n_votes=30000, gamma=args.simulate, seed=7)
        votes = pd.DataFrame({"model_a": left, "model_b": right, "left_won": y, "judge_hash": voter})
        result["counts"] = {"simulated_gamma": args.simulate, "decisive": len(votes)}
    else:
        raw = pd.read_csv(args.data, low_memory=False)
        if "anony" in raw.columns:
            raw = raw[raw["anony"] == True]  # sides only hidden in anon battles
        one_sided, result["balance"] = balance_check(raw)
        kept = raw[~raw["model_a"].isin(one_sided) & ~raw["model_b"].isin(one_sided)]
        tie_share = float(kept["winner"].astype(str).str.startswith("tie").mean())
        votes = kept[kept["winner"].isin(["model_a", "model_b"])].copy()
        votes["left_won"] = (votes["winner"] == "model_a").astype(int)
        result["counts"] = {"all_rows": len(raw), "after_dropping_one_sided_models": len(kept),
                            "decisive": len(votes), "tie_share": tie_share,
                            "voters": int(votes["judge_hash"].nunique())}

    main_fit = fit_side_effect(votes["model_a"], votes["model_b"], votes["left_won"])
    se = main_fit["se"]
    n_left = int(votes["left_won"].sum())
    result["main"] = {k: main_fit[k] for k in ("n_votes", "n_models", "left_win_share", "gamma", "se", "ci_low",
                                               "ci_high", "p_left_when_even", "lrt", "p_value")}
    result["main"]["binomial_p_value"] = float(stats.binomtest(n_left, len(votes), 0.5).pvalue)
    result["main"]["detectable_effect_points"] = float(100 * (1 / (1 + np.exp(-2.8 * se)) - 0.5))
    low, high = cluster_bootstrap_gamma(votes["model_a"], votes["model_b"], votes["left_won"],
                                        votes["judge_hash"], n_boot=args.boot)
    result["main"]["ci_resampling_voters"] = [low, high]

    with_side = pd.Series(main_fit["strengths"])
    without = pd.Series(main_fit["strengths_no_side"])[with_side.index]
    rank_shift = (with_side.rank(ascending=False) - without.rank(ascending=False)).abs()
    result["ranking"] = {"spearman": float(stats.spearmanr(with_side, without)[0]),
                         "models_changing_rank": int((rank_shift > 0).sum()),
                         "largest_rank_change": int(rank_shift.max()),
                         "largest_strength_change": float((with_side - without).abs().max())}

    rows = [row("all decisive votes", votes)]
    if "judge_hash" in votes.columns:
        per_voter = votes["judge_hash"].map(votes["judge_hash"].value_counts()).to_numpy()
        for label, lo, hi in [("voters with 1 vote", 1, 1), ("voters with 2-4 votes", 2, 4),
                              ("voters with 5-19 votes", 5, 19), ("voters with 20+ votes", 20, 10**9)]:
            rows.append(row(label, votes[(per_voter >= lo) & (per_voter <= hi)]))
    if "tstamp" in votes.columns:
        month = pd.to_datetime(votes["tstamp"], unit="s").dt.to_period("M")
        for m in sorted(month.unique()):
            rows.append(row(f"month {m}", votes[month == m]))
    if "turn" in votes.columns:
        rows.append(row("first turn only", votes[votes["turn"] == 1]))
        rows.append(row("multi-turn", votes[votes["turn"] > 1]))
    if "is_code" in votes.columns:
        rows.append(row("code prompts", votes[votes["is_code"] == True]))
        rows.append(row("non-code prompts", votes[votes["is_code"] == False]))
    if "language" in votes.columns:
        rows.append(row("English", votes[votes["language"] == "English"]))
        rows.append(row("other languages", votes[votes["language"] != "English"]))
    table = pd.DataFrame([r for r in rows if r is not None])

    if LEN_A in votes.columns and LEN_B in votes.columns:
        fr = votes[(votes[LEN_A] > 0) & (votes[LEN_B] > 0)]
        log_ratio = np.log(fr[LEN_A].to_numpy(float) / fr[LEN_B].to_numpy(float))
        fit = fit_side_effect(fr["model_a"], fr["model_b"], fr["left_won"], covariates={"log_length_ratio": log_ratio})
        b = fit["covariates"]["log_length_ratio"]
        result["length"] = {
            "gamma_holding_length_fixed": fit["gamma"], "gamma_ci": [fit["ci_low"], fit["ci_high"]],
            "p_value": fit["p_value"], "length_coef": b["coef"],
            "length_ci": [b["coef"] - 1.96 * b["se"], b["coef"] + 1.96 * b["se"]],
            "p_win_if_twice_as_long": float(1 / (1 + np.exp(-b["coef"] * np.log(2)))),
            "length_difference_equal_to_side_effect_pct": float(100 * (np.exp(abs(fit["gamma"]) / b["coef"]) - 1)),
        }

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/side_effect_by_group.csv", index=False)
    with open("results/summary.json", "w") as f:
        json.dump(result, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(table) + 1.6))
    ypos = np.arange(len(table))[::-1]
    points = 100 * (table["p_left_when_even"] - 0.5)
    lo = 100 * (1 / (1 + np.exp(-table["ci_low"])) - 0.5)
    hi = 100 * (1 / (1 + np.exp(-table["ci_high"])) - 0.5)
    ax.errorbar(points, ypos, xerr=[points - lo, hi - points], fmt="o", color="#1f4e79", ecolor="#7f9fbf", capsize=3)
    ax.axvline(0.0, color="grey", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(table["group"])
    ax.set_xlabel("left-side advantage between evenly matched models (percentage points, 95% interval)")
    ax.set_title("Does the answer on the left win more often than it should?")
    fig.tight_layout()
    fig.savefig("results/side_effect_by_group.png", dpi=200)

    pd.set_option("display.width", 200)
    print(json.dumps(result, indent=2))
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
