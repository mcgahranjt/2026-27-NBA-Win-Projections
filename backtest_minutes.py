"""
Minutes allocation backtest

Checks how nba_player_props.py should fit each team's minutes to 240 when its players' usual
minutes add up to more than that. Every game in the opening weeks of a season is projected the way
the props script does it (minutes_baseline: last season's minutes per game blended with earlier
games that season), then fit three ways using allocate() with different DEPTH_WEIGHT settings:
  even   - DEPTH_WEIGHT 0: everyone cut by the same share
  depth  - DEPTH_WEIGHT 1: Version B's depth chart; the overflow comes off the end of the bench
  blend  - the current setting in nba_player_props.py
Projected minutes are compared with actual minutes for players who played.

Who was out: there are no historical injury reports, so a player with a 15+ minute role who missed
the game and the team's games on either side of it is treated as out. Rosters are the players who
appeared for each team in the window. Both rules apply the same way to every method.

Usage:
  python3 backtest_minutes.py            # 2025-26 (the test behind the current setting)
  python3 backtest_minutes.py 2026-27    # this season, once a few weeks of games are in
"""

import os
import sys

import numpy as np
import pandas as pd

import nba_player_props as P
import nba_wins_version_b as V
from nba_wins_version_b import name_key

WINDOWS = [14, 28]          # days from opening night
OUT_ROLE_MPG = 15           # missing three straight games with at least this role = treated as out
HEAVY_OVERFLOW = 0.90       # games where everyone would be cut to 90% or less
METHODS = {"even": 0.0, "depth": 1.0, "blend": P.DEPTH_WEIGHT}
ROOKIE_MPG = V.ROOKIE_MINUTES / V.EXPECTED_GAMES


def game_log(season):
    path = os.path.join(P.S.DATA_DIR, f"player_game_log_{season}.csv")
    if not os.path.exists(path):
        from nba_api.stats.endpoints import leaguegamelog
        log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="P",
                                          season_type_all_star="Regular Season", timeout=60).get_data_frames()[0]
        log[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN"] + P.STATS].to_csv(path, index=False)
    log = pd.read_csv(path)
    log["GAME_DATE"] = pd.to_datetime(log.GAME_DATE)
    log["GAME_ID"] = log.GAME_ID.astype(str)
    log["MIN"] = pd.to_numeric(log.MIN, errors="coerce").fillna(0)
    return log[log.MIN > 0]


def run(played, prev1, days):
    start = played.GAME_DATE.min()
    win = played[played.GAME_DATE < start + pd.Timedelta(days=days)]
    roster = win.drop_duplicates(["PLAYER_ID", "TEAM_ID"])[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]]
    roster = roster.assign(key=roster.PLAYER_NAME.map(name_key))
    # Version B-style prior: last season's minutes per game, rookies 600 minutes over 68 games
    last = prev1.assign(mpg=prev1.MIN / prev1.GP.clip(lower=1)).groupby("PLAYER_ID").mpg.max()
    prior = pd.Series(roster.PLAYER_ID.map(last).fillna(ROOKIE_MPG).to_numpy(), index=roster.key)
    prior = prior[~prior.index.duplicated()]

    team_games = win.drop_duplicates(["TEAM_ID", "GAME_ID"]).sort_values("GAME_DATE")[["TEAM_ID", "GAME_ID", "GAME_DATE"]]
    who = win.groupby(["GAME_ID", "TEAM_ID"]).PLAYER_ID.apply(set)
    mins = win.set_index(["GAME_ID", "PLAYER_ID"]).MIN
    # on the roster from opening week, or from his first appearance if he joined (or returned) later
    opener = team_games.groupby("TEAM_ID").GAME_DATE.min()
    fa = win.groupby(["TEAM_ID", "PLAYER_ID"]).GAME_DATE.min().reset_index()
    fa["since"] = np.where(fa.GAME_DATE <= fa.TEAM_ID.map(opener) + pd.Timedelta(days=7),
                           fa.TEAM_ID.map(opener), fa.GAME_DATE)
    since = fa.set_index(["TEAM_ID", "PLAYER_ID"]).since

    rows = []
    orig = P.DEPTH_WEIGHT
    try:
        for d in sorted(team_games.GAME_DATE.unique()):
            base = P.minutes_baseline(roster, played[played.GAME_DATE < d], prev1, prior, d)
            for tid, gid in team_games[team_games.GAME_DATE == d][["TEAM_ID", "GAME_ID"]].itertuples(index=False):
                tg = team_games[team_games.TEAM_ID == tid].GAME_ID.tolist()
                i = tg.index(gid)
                near = set().union(*[who.get((tg[j], tid), set()) for j in (i - 1, i + 1) if 0 <= j < len(tg)])
                here = who.get((gid, tid), set())
                t = base[base.TEAM_ID == tid].copy()
                joined = [since.get((tid, p), pd.Timestamp.max) <= d for p in t.PLAYER_ID]
                missed = ~t.PLAYER_ID.isin(here | near)
                t = t[np.array(joined) & ~(missed & (t.prior_mpg >= OUT_ROLE_MPG))]
                if t.empty:
                    continue
                for name, w in METHODS.items():
                    P.DEPTH_WEIGHT = w
                    t[name] = P.allocate(t)
                t["actual"] = [mins.get((gid, p), 0) for p in t.PLAYER_ID]
                t["played"] = t.PLAYER_ID.isin(here)
                t["scale"] = P.TEAM_MINUTES / (t.baseline * t.part).sum()
                t["game_id"], t["team"] = gid, tid
                rows.append(t)
    finally:
        P.DEPTH_WEIGHT = orig
    return pd.concat(rows, ignore_index=True)


def table(pl):
    tiers = pd.cut(pl.baseline, [0, 10, 20, 28, 48], labels=["<10", "10-20", "20-28", "28+"], right=False)
    out = {}
    for name, x in list(pl.groupby(tiers, observed=True)) + [("all", pl)]:
        out[name] = {"n": len(x), "actual": x.actual.mean()}
        for m in METHODS:
            out[name][f"{m}_bias"] = (x[m] - x.actual).mean()
            out[name][f"{m}_MAE"] = (x[m] - x.actual).abs().mean()
    return pd.DataFrame(out).T.round(2)


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    start_year = int(season[:4])
    played = game_log(season)
    prev1 = P.prev_season_totals(start_year - 1)
    print(f"Minutes allocation backtest, {season} (current DEPTH_WEIGHT = {P.DEPTH_WEIGHT})")
    print("Rows are players grouped by their usual minutes per game; bias = projected minus actual.")
    for days in WINDOWS:
        r = run(played, prev1, days)
        pl = r[r.played]
        tg = r.drop_duplicates(["game_id", "team"])
        print(f"\nFirst {days} days: {len(tg)} team-games, {len(pl)} player-games. Share of minutes kept "
              f"before fitting: median {tg.scale.median():.2f} (10th-90th {tg.scale.quantile(.1):.2f}-"
              f"{tg.scale.quantile(.9):.2f})")
        print(table(pl).to_string())
        if days == WINDOWS[-1]:
            heavy = pl[pl.scale <= HEAVY_OVERFLOW]
            n = heavy.groupby(["game_id", "team"]).ngroups
            print(f"\nGames where everyone would be cut to {HEAVY_OVERFLOW:.0%} or less ({n} team-games):")
            print(table(heavy).to_string())


if __name__ == "__main__":
    main()
