"""
NBA win projection - Version A (team-level regression)

Predicts each team's wins for the upcoming season from:
  1. Last season's net rating (points per 100 possessions, scored minus allowed)
  2. Returning minutes share (share of last season's team minutes played by
     players who are still on the roster)
  3. Net rating x returning share (last year's rating matters more when the roster stayed together)
  4. Age, tested two ways - the script backtests both and uses whichever is more accurate:
       a. roster_age: minutes-weighted average age of the upcoming roster
       b. vet_min_share: share of minutes going to players 31 or older (aging risk)

Steps:
  1. Pull team and player stats from NBA.com (via nba_api) and cache to CSV
  2. Build one row per team-season: features from season s, target = wins in s+1
  3. Backtest against two simple baselines
  4. Fit on all history and project the target season

Usage:
  pip install nba_api pandas scikit-learn matplotlib
  python nba_wins_version_a.py

Re-running uses the cached CSVs in ./data. Delete them to refresh.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

FIRST_SEASON = 2010          # 2010-11
LAST_COMPLETED = 2025        # 2025-26 is the most recent finished season
TARGET = LAST_COMPLETED + 1  # project 2026-27
GAMES = 82
BASE_FEATURES = ["prev_net_rating", "returning_min_share", "net_x_returning"]
VARIANTS = {
    "avg_age": BASE_FEATURES + ["roster_age"],
    "vet_share": BASE_FEATURES + ["vet_min_share"],
}
VET_AGE = 31
DATA_DIR = "data"
OUT_DIR = "output"


def season_str(year):
    """2025 -> '2025-26'"""
    return f"{year}-{str(year + 1)[-2:]}"


def call_api(fn, retries=4, **kwargs):
    """Call an nba_api endpoint with polite pauses and retries (stats.nba.com is flaky)."""
    for attempt in range(retries):
        try:
            time.sleep(0.8)
            return fn(timeout=60, **kwargs).get_data_frames()[0]
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  request failed ({e.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {fn.__name__} {kwargs}")


# ---------------------------------------------------------------- 1. data pull
def fetch_team_seasons():
    path = os.path.join(DATA_DIR, "team_seasons.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from nba_api.stats.endpoints import leaguedashteamstats
    rows = []
    for y in range(FIRST_SEASON, LAST_COMPLETED + 1):
        print(f"team stats {season_str(y)}")
        df = call_api(leaguedashteamstats.LeagueDashTeamStats,
                      season=season_str(y),
                      measure_type_detailed_defense="Advanced")
        df = df[["TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT", "NET_RATING"]].copy()
        df["season"] = y
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def fetch_player_seasons():
    """Player totals per season. For traded players, NBA.com lists the team they ended the season with."""
    path = os.path.join(DATA_DIR, "player_seasons.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from nba_api.stats.endpoints import leaguedashplayerstats
    rows = []
    for y in range(FIRST_SEASON, LAST_COMPLETED + 1):
        print(f"player stats {season_str(y)}")
        df = call_api(leaguedashplayerstats.LeagueDashPlayerStats,
                      season=season_str(y),
                      per_mode_detailed="Totals")
        df = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "AGE", "MIN"]].copy()
        df["season"] = y
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def fetch_target_rosters(team_ids):
    """Current rosters for the season being projected."""
    path = os.path.join(DATA_DIR, f"rosters_{TARGET}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from nba_api.stats.endpoints import commonteamroster
    rows = []
    for tid in team_ids:
        print(f"roster {tid} {season_str(TARGET)}")
        df = call_api(commonteamroster.CommonTeamRoster,
                      team_id=int(tid), season=season_str(TARGET))
        rows.append(pd.DataFrame({"PLAYER_ID": df["PLAYER_ID"], "TEAM_ID": tid}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


# ------------------------------------------------------------ 2. features
def build_features(teams, players, next_team_of, season):
    """
    Features for each team going into season+1.
    next_team_of: DataFrame [PLAYER_ID, TEAM_ID] giving each player's team in season+1.
    Only season-level data from `season` is used, so nothing from the predicted season leaks in
    (apart from who is on the roster, which is known before opening night).
    """
    prev = players[players.season == season]
    team_min = prev.groupby("TEAM_ID")["MIN"].sum().rename("team_min")

    # returning minutes: last-season minutes by players who stay with the same team
    merged = prev.merge(next_team_of.rename(columns={"TEAM_ID": "NEXT_TEAM_ID"}),
                        on="PLAYER_ID", how="left")
    stayed = merged[merged.TEAM_ID == merged.NEXT_TEAM_ID]
    returning = stayed.groupby("TEAM_ID")["MIN"].sum().rename("ret_min")

    # roster age next season: last-season minutes as weights, age + 1
    nxt = next_team_of.merge(prev[["PLAYER_ID", "AGE", "MIN"]], on="PLAYER_ID", how="left")
    nxt["AGE"] = nxt["AGE"] + 1
    nxt["MIN"] = nxt["MIN"].fillna(0)
    league_avg_age = (prev["AGE"] * prev["MIN"]).sum() / prev["MIN"].sum()
    nxt["AGE"] = nxt["AGE"].fillna(league_avg_age)  # rookies / players with no prior season
    nxt["w"] = nxt["MIN"].clip(lower=100)            # give newcomers a small weight instead of zero
    age = nxt.groupby("TEAM_ID").apply(lambda g: np.average(g["AGE"], weights=g["w"]),
                                       include_groups=False).rename("roster_age")
    nxt["vet_w"] = np.where(nxt["AGE"] >= VET_AGE, nxt["w"], 0)
    vets = (nxt.groupby("TEAM_ID")["vet_w"].sum() / nxt.groupby("TEAM_ID")["w"].sum()).rename("vet_min_share")

    t = teams[teams.season == season].set_index("TEAM_ID")
    f = pd.DataFrame({
        "TEAM_NAME": t["TEAM_NAME"],
        "prev_net_rating": t["NET_RATING"],
        "prev_wins_82": t["W_PCT"] * GAMES,
    })
    f = f.join(team_min).join(returning).join(age).join(vets)
    f["ret_min"] = f["ret_min"].fillna(0)
    f["returning_min_share"] = f["ret_min"] / f["team_min"]
    # last year's rating says more about a team that kept its roster together
    f["net_x_returning"] = f["prev_net_rating"] * f["returning_min_share"]
    f["season"] = season + 1
    return f.reset_index()[["TEAM_ID", "TEAM_NAME", "season", "prev_wins_82"]
                           + BASE_FEATURES + ["roster_age", "vet_min_share"]]


def build_history(teams, players):
    rows = []
    for y in range(FIRST_SEASON, LAST_COMPLETED):
        next_team_of = players.loc[players.season == y + 1, ["PLAYER_ID", "TEAM_ID"]]
        f = build_features(teams, players, next_team_of, y)
        target = teams[teams.season == y + 1][["TEAM_ID", "W_PCT"]]
        f = f.merge(target, on="TEAM_ID")
        f["wins_82"] = f["W_PCT"] * GAMES  # scale shortened seasons (2011-12, 2019-20, 2020-21) to 82 games
        rows.append(f.drop(columns="W_PCT"))
    return pd.concat(rows, ignore_index=True)


def normalize_to_league(pred, games=GAMES, n_teams=30):
    """Every game has one winner, so league wins must total games * teams / 2."""
    total = games * n_teams / 2
    return pred + (total - pred.sum()) / len(pred)


# ---------------------------------------------------------- 3. backtest
def backtest(hist, test_seasons):
    results = []
    for s in test_seasons:
        train = hist[hist.season < s]
        test = hist[hist.season == s].copy()
        row = {"season": season_str(s)}
        for name, feats in VARIANTS.items():
            model = LinearRegression().fit(train[feats], train["wins_82"])
            pred = normalize_to_league(model.predict(test[feats]))
            row[f"model_{name}_MAE"] = mean_absolute_error(test["wins_82"], pred)
        results.append({**row,
            "baseline_41_MAE": mean_absolute_error(test["wins_82"], np.full(len(test), 41.0)),
            "baseline_last_year_MAE": mean_absolute_error(test["wins_82"], test["prev_wins_82"]),
        })
    return pd.DataFrame(results)


# ------------------------------------------------------------ 4. project
def chart(proj, path):
    proj = proj.sort_values("projected_wins")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(proj["TEAM_NAME"], proj["projected_wins"], color="#3b6ea5")
    ax.axvline(41, color="#888", lw=1, ls="--")
    for i, v in enumerate(proj["projected_wins"]):
        ax.text(v + 0.4, i, f"{v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("Projected wins")
    ax.set_title(f"{season_str(TARGET)} projected wins (Version A)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    teams = fetch_team_seasons()
    players = fetch_player_seasons()
    hist = build_history(teams, players)
    hist.to_csv(os.path.join(OUT_DIR, "training_data.csv"), index=False)

    bt = backtest(hist, test_seasons=range(LAST_COMPLETED - 4, LAST_COMPLETED + 1))
    print("\nBacktest (mean absolute error in wins, lower is better):")
    print(bt.round(2).to_string(index=False))
    print("\nAverage:")
    print(bt.drop(columns="season").mean().round(2).to_string())
    bt.to_csv(os.path.join(OUT_DIR, "backtest.csv"), index=False)

    means = bt.drop(columns="season").mean()
    best = min(VARIANTS, key=lambda n: means[f"model_{n}_MAE"])
    FEATURES = VARIANTS[best]
    print(f"\nUsing '{best}' for projections (lowest average error).")
    for name, feats in VARIANTS.items():
        m = LinearRegression().fit(hist[feats], hist["wins_82"])
        print(f"Model {name}: wins = {m.intercept_:.1f}"
              + "".join(f" + {c:.2f} x {f}" for c, f in zip(m.coef_, feats)))
    model = LinearRegression().fit(hist[FEATURES], hist["wins_82"])

    team_ids = teams.loc[teams.season == LAST_COMPLETED, "TEAM_ID"].unique()
    rosters = fetch_target_rosters(team_ids)
    feats = build_features(teams, players, rosters, LAST_COMPLETED)
    feats["projected_wins"] = normalize_to_league(model.predict(feats[FEATURES]))
    proj = feats.sort_values("projected_wins", ascending=False)
    proj["projected_wins"] = proj["projected_wins"].round(1)
    cols = ["TEAM_NAME", "projected_wins", "prev_wins_82"] + FEATURES
    proj[cols].to_csv(os.path.join(OUT_DIR, f"projections_{season_str(TARGET)}.csv"), index=False)
    print(f"\n{season_str(TARGET)} projections:")
    print(proj[cols].round(2).to_string(index=False))

    chart(proj, os.path.join(OUT_DIR, f"projections_{season_str(TARGET)}.png"))
    print(f"\nSaved results to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
