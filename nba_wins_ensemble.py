"""
NBA win projection - A+B average

Version A (team-level) and Version B (player-level) make different kinds of mistakes, so an
average of the two may beat either one alone. This script:

  1. Makes rolling predictions from both models for every season 2015-16 through 2025-26
     (each season predicted using only earlier seasons)
  2. Picks the weight on B (0, 25, 50, 75 or 100%) using 2015-16 to 2020-21 only
  3. Reports error on the test seasons 2021-22 to 2025-26 for A, B, a plain 50/50 average,
     and the tuned weight
  4. Projects 2026-27 with all of them

Needs nba_wins_version_a.py and nba_wins_version_b.py in the same folder, plus the data folder
both of them created. Nothing is downloaded - it all runs from the cached files.

Usage:
  python3 nba_wins_ensemble.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import nba_wins_version_a as A
import nba_wins_version_b as B

OUT_DIR = "output_ensemble"
FEATS_A = A.VARIANTS["avg_age"]
FEATS_B = ["team_rating"]
WEIGHT_OPTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]  # weight on Version B
TUNE_SEASONS = range(2015, 2021)
TEST_SEASONS = range(2021, 2026)


def rolling_predictions(hist, feats, seasons, label):
    """Predict each season using a model trained only on the seasons before it."""
    out = []
    for s in seasons:
        tr = hist[hist.season < s].dropna(subset=feats + ["wins_82"])
        te = hist[hist.season == s].copy()
        m = LinearRegression().fit(tr[feats], tr.wins_82)
        te[label] = A.normalize_to_league(m.predict(te[feats]))
        out.append(te[["TEAM_ID", "season", "wins_82", "prev_wins_82", label]])
    return pd.concat(out, ignore_index=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seasons = range(min(TUNE_SEASONS), max(TEST_SEASONS) + 1)

    # Version A
    teams = A.fetch_team_seasons()
    players_nba = A.fetch_player_seasons()
    hist_a = A.build_history(teams, players_nba)
    pred_a = rolling_predictions(hist_a, FEATS_A, seasons, "pred_A")

    # Version B
    ps = B.player_seasons_br()
    curves = {t: B.aging_curve(ps, t - 1) for t in range(B.FIRST_TARGET, B.TARGET + 1)}
    prior_minutes, _ = B.tune_prior_minutes(ps, teams, curves)
    hist_b = B.build_history(ps, teams, prior_minutes, curves)
    pred_b = rolling_predictions(hist_b, FEATS_B, seasons, "pred_B")

    df = pred_a.merge(pred_b[["TEAM_ID", "season", "pred_B"]], on=["TEAM_ID", "season"])
    if df.groupby("season").size().min() < 30:
        print("Warning: some teams didn't line up between A and B")

    # pick the weight on B using the tuning seasons only
    tune = df[df.season.isin(TUNE_SEASONS)]
    tune_err = {w: mean_absolute_error(tune.wins_82, (1 - w) * tune.pred_A + w * tune.pred_B)
                for w in WEIGHT_OPTIONS}
    best_w = min(tune_err, key=tune_err.get)
    print("Weight on B, tuned on 2015-16 to 2020-21: "
          + ", ".join(f"{int(w * 100)}% = {e:.2f}" for w, e in tune_err.items())
          + f" -> using {int(best_w * 100)}%")

    df["pred_avg"] = 0.5 * df.pred_A + 0.5 * df.pred_B
    df["pred_tuned"] = (1 - best_w) * df.pred_A + best_w * df.pred_B

    test = df[df.season.isin(TEST_SEASONS)]
    rows = []
    for s, g in test.groupby("season"):
        rows.append({
            "season": A.season_str(s),
            "A_MAE": mean_absolute_error(g.wins_82, g.pred_A),
            "B_MAE": mean_absolute_error(g.wins_82, g.pred_B),
            "avg_50_50_MAE": mean_absolute_error(g.wins_82, g.pred_avg),
            f"tuned_{int(best_w * 100)}pct_B_MAE": mean_absolute_error(g.wins_82, g.pred_tuned),
            "baseline_last_year_MAE": mean_absolute_error(g.wins_82, g.prev_wins_82),
        })
    bt = pd.DataFrame(rows)
    print("\nBacktest (mean absolute error in wins, lower is better):")
    print(bt.round(2).to_string(index=False))
    print("\nAverage:")
    print(bt.drop(columns="season").mean().round(2).to_string())
    bt.to_csv(os.path.join(OUT_DIR, "backtest_ensemble.csv"), index=False)
    print(f"\nCorrelation between A and B errors: "
          f"{np.corrcoef(test.pred_A - test.wins_82, test.pred_B - test.wins_82)[0, 1]:.2f} "
          "(lower means averaging helps more)")

    # 2026-27
    model_a = LinearRegression().fit(hist_a[FEATS_A], hist_a.wins_82)
    feats_a = A.build_features(teams, players_nba, A.fetch_target_rosters(
        teams.loc[teams.season == A.LAST_COMPLETED, "TEAM_ID"].unique()), A.LAST_COMPLETED)
    feats_a["A"] = A.normalize_to_league(model_a.predict(feats_a[FEATS_A]))

    model_b = LinearRegression().fit(hist_b[FEATS_B], hist_b.wins_82)
    rosters = B.fetch_target_rosters_named(teams.loc[teams.season == A.LAST_COMPLETED, "TEAM_ID"].unique())
    rosters["key"] = rosters.player.map(B.name_key)
    proj = B.project_players(ps, B.TARGET, curves[B.TARGET], prior_minutes)
    team_b, _ = B.team_ratings(rosters[["key", "player", "TEAM_ID"]], proj)
    feats_b = team_b.reset_index()
    feats_b["B"] = A.normalize_to_league(model_b.predict(feats_b[FEATS_B]))

    out = feats_a[["TEAM_ID", "TEAM_NAME", "prev_wins_82", "A"]].merge(feats_b[["TEAM_ID", "B"]], on="TEAM_ID")
    out["average_50_50"] = 0.5 * out.A + 0.5 * out.B
    out["tuned"] = (1 - best_w) * out.A + best_w * out.B
    out["A_minus_B"] = out.A - out.B
    out = out.sort_values("tuned", ascending=False).drop(columns="TEAM_ID").round(1)
    out = out.rename(columns={"prev_wins_82": "2025-26 wins", "tuned": f"tuned_{int(best_w * 100)}pct_B"})
    out.to_csv(os.path.join(OUT_DIR, f"projections_ensemble_{A.season_str(A.TARGET)}.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\n{A.season_str(A.TARGET)} projections:")
    print(out.to_string(index=False))

    # chart: A and B side by side for each team, sorted by the tuned projection
    col = out.columns[out.columns.str.startswith("tuned")][0]
    o = out.sort_values(col)
    y = np.arange(len(o))
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.barh(y, o[col], color="#3b6ea5", alpha=0.35, label=f"Combined ({col.split('_')[1]} B)")
    ax.scatter(o.A, y, color="#c0504d", s=18, label="Version A", zorder=3)
    ax.scatter(o.B, y, color="#1f3b5c", s=18, marker="D", label="Version B", zorder=3)
    ax.set_yticks(y, o.TEAM_NAME, fontsize=8)
    ax.axvline(41, color="#888", lw=1, ls="--")
    ax.set_xlabel("Projected wins")
    ax.set_title(f"{A.season_str(A.TARGET)} projected wins: A, B and combined")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"projections_ensemble_{A.season_str(A.TARGET)}.png"), dpi=150)
    print(f"\nSaved results to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
