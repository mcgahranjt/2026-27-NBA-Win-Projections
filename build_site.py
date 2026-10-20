"""
Projections website

Builds a one-page website from the files the other scripts already write (nothing is re-estimated):
  Tonight      - the day's games (or the next game day) from output_power/game_projections.csv,
                 with each team's projected players from output_props/
  Players      - every projected player's stats and 10th-90th percentile ranges
  Rankings     - output_power/power_rankings.csv and team_ratings.csv
  Season Odds  - output_sim/sim_summary.csv, win_distribution.csv and tracker_history.csv
The page itself is site_template.html; this script fills in the data and writes site/index.html.

It reads only local output files. It never reads the Google Sheet.

With --publish, the page is copied into a local clone of the GitHub Pages repository
(SITE_REPO below) and pushed, which updates the public site a minute or two later. Until that clone
exists, --publish only builds.

Usage:
  python3 build_site.py             # build site/index.html (open it in a browser to check)
  python3 build_site.py --publish   # build, then commit and push to GitHub Pages
"""

import datetime as dt
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

import nba_player_props as P
import nba_power_ratings as PR
import nba_season_sim as S

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "site_template.html")
OUT_DIR = os.path.join(HERE, "site")
SITE_REPO = os.path.expanduser("~/Documents/NBAProj/nba-projections-site")
PLAYER_COLS = ["Player", "Team", "Opp", "Home/Away", "Status", "MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK",
               "TOV", "PRA", "P+R", "P+A", "R+A", "DD%"]


def read(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw) if os.path.exists(path) else None


def clean(records):
    """Make DataFrame records JSON-safe (NaN -> None, numpy -> Python)."""
    out = []
    for r in records:
        out.append({k: (None if isinstance(v, float) and pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                    for k, v in r.items()})
    return out


def short_name(team):
    return "Trail Blazers" if team.endswith("Trail Blazers") else team.split()[-1]


def gather():
    teams = S.load_teams()
    abbr = {name: P.ABBR[tid] for tid, name in zip(teams.TEAM_ID, teams.TEAM_NAME)}
    today = dt.date.today().isoformat()

    # games: today if there are any, otherwise the next day with games
    gp = read(os.path.join(PR.OUT_DIR, "game_projections.csv"), dtype={"game_id": str})
    games, game_date = [], today
    if gp is not None and len(gp):
        upcoming = gp[gp.date >= today]
        if len(upcoming):
            game_date = upcoming.date.min()
            for r in upcoming[upcoming.date == game_date].itertuples():
                games.append({"id": r.game_id, "away": abbr[r.away], "home": abbr[r.home], "awayName": r.away,
                              "homeName": r.home, "margin": r.proj_margin, "total": r.proj_total, "hwp": r.home_win_prob})

    # player projections and ranges for that day, if the props script has run for it
    players, ranges = [], {}
    pj = read(os.path.join(P.OUT_DIR, f"player_projections_{game_date}.csv"))
    rg = read(os.path.join(P.OUT_DIR, f"player_ranges_{game_date}.csv"))
    if pj is not None:
        pj = pj[PLAYER_COLS].copy()
        pj["Status"] = pj.Status.fillna("")
        pj["key"] = pj.Player + "|" + pj.Team
        players = clean(pj.to_dict("records"))
        if rg is not None:
            for r in rg.itertuples():
                ranges.setdefault(f"{r.Player}|{r.Team}", {})[r.Stat] = [r.Mean, r.P10, r.P25, r.Median, r.P75, r.P90]

    # rankings
    pr = read(os.path.join(PR.OUT_DIR, "power_rankings.csv"))
    tr = read(os.path.join(PR.OUT_DIR, "team_ratings.csv"))
    lv = read(os.path.join(PR.OUT_DIR, "league_levels.csv"))
    rank = []
    if pr is not None and tr is not None:
        m = pr.merge(tr[["TEAM_NAME", "off_rtg", "def_rtg", "pace", "off_rank", "def_rank"]], on="TEAM_NAME", how="left")
        for r in m.itertuples():
            rank.append({"rank": int(r.rank), "chg": None if pd.isna(r.rank_change) else int(r.rank_change),
                         "team": r.TEAM_NAME, "ab": abbr[r.TEAM_NAME], "W": int(r.W), "L": int(r.L),
                         "rating": r.rating, "w82": r.pace_82, "off": r.off_rtg, "def": r.def_rtg, "pace": r.pace,
                         "offr": int(r.off_rank), "defr": int(r.def_rank)})

    # season odds
    sim_dir = S.OUT_DIR
    sm = read(os.path.join(sim_dir, "sim_summary.csv"))
    wd = read(os.path.join(sim_dir, "win_distribution.csv"))
    th = read(os.path.join(sim_dir, "tracker_history.csv"))
    odds, dist, trend = [], {}, {}
    if sm is not None:
        for r in sm.itertuples():
            odds.append({"ab": abbr[r.TEAM_NAME], "team": r.TEAM_NAME, "short": short_name(r.TEAM_NAME), "conf": r.conf,
                         "W": int(r.W), "L": int(r.L), "pre": r.preseason_proj, "mean": r.mean_wins,
                         "p10": int(r.p10_wins), "p90": int(r.p90_wins), "rating": r.rating,
                         "top6": r.top6_pct, "playin": r.playin_pct, "playoffs": r.playoffs_pct, "r2": r.round2_pct,
                         "cf": r.conf_finals_pct, "finals": r.finals_pct, "title": r.title_pct,
                         "seed": int(r.most_likely_seed)})
    if wd is not None:
        for team, g in wd.groupby("TEAM_NAME"):
            dist[abbr[team]] = [[int(w), round(float(p), 2)] for w, p in zip(g.wins, g.pct)]
    if th is not None:
        for team, g in th.sort_values("snapshot_date").groupby("TEAM_NAME"):
            trend[abbr[team]] = [[d, float(w)] for d, w in zip(g.snapshot_date, g.mean_wins)]
    sim_as_of = th.snapshot_date.max() if th is not None and len(th) else ""

    return {"today": today, "gameDate": game_date, "built": dt.datetime.now().strftime("%b %-d, %-I:%M %p"),
            "ratingsAsOf": lv.as_of.iloc[0] if lv is not None else "", "simAsOf": sim_as_of, "simRuns": S.N_SIMS,
            "games": games, "players": players, "ranges": ranges, "rank": rank,
            "odds": odds, "dist": dist, "trend": trend}


def build():
    data = gather()
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__DATA__", payload)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Built {path}: {len(data['games'])} games on {data['gameDate']}, {len(data['players'])} players, "
          f"{len(data['rank'])} teams ranked, {len(data['odds'])} teams in the season odds.")
    return path


def git(*args):
    return subprocess.run(["git", "-C", SITE_REPO] + list(args), capture_output=True, text=True)


def publish(path):
    if not os.path.isdir(os.path.join(SITE_REPO, ".git")):
        print(f"Not published: no GitHub Pages repository at {SITE_REPO} yet.")
        return True
    shutil.copy(path, os.path.join(SITE_REPO, "index.html"))
    git("add", "index.html")
    if git("diff", "--cached", "--quiet").returncode == 0:
        print("Website unchanged since the last publish.")
        return True
    msg = f"Update projections {dt.datetime.now():%Y-%m-%d %H:%M}"
    c = git("commit", "-m", msg)
    p = git("push", "origin", "HEAD")
    if c.returncode or p.returncode:
        print("Couldn't publish the website:\n" + (c.stdout + c.stderr + p.stderr).strip()[-600:])
        return False
    print("Published to GitHub Pages (live in a minute or two).")
    return True


def main():
    path = build()
    if "--publish" in sys.argv and not publish(path):
        sys.exit(1)


if __name__ == "__main__":
    main()
