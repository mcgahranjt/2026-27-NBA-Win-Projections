"""
NBA season simulator - Version C

Simulates the 2026-27 season thousands of times, starting from the combined model's
projections, to get a range of outcomes and playoff odds for every team.

Run it before the season for preseason odds. Run it again during the season (weekly works well)
and it will:
  - pull completed results from NBA.com's schedule feed
  - update each team's rating from its actual point differential so far
  - simulate only the remaining games
  - add a dated snapshot to output_sim/tracker_history.csv for the dashboard

How it works:
  - Team rating (points better than average per game) = (projected wins - 41) / 2.7
  - Each game: home team wins with probability Phi((home rating - away rating + home court) / 12)
  - Each simulated season also draws a random error for every team's rating, sized from the
    combined model's backtest error, so the ranges reflect how wrong preseason projections can be
  - Standings -> play-in (seeds 7-10) -> four rounds of best-of-seven playoffs

Needs output_ensemble/projections_ensemble_2026-27.csv and data/team_seasons.csv.

Usage:
  python3 nba_season_sim.py
"""

import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

SEASON = "2026-27"
N_SIMS = 10000
SEED = 2027

WINS_PER_POINT = 2.7   # wins added by one point of per-game margin over a season
GAME_SD = 12.0         # standard deviation of an NBA game's final margin, in points
HOME_COURT = 2.3       # home-court advantage in points
BACKTEST_MAE = 6.95    # combined model's average preseason error, in wins
PRIOR_GAMES = 25       # in-season: the preseason rating counts as much as this many games of results

DATA_DIR = "data"
OUT_DIR = "output_sim"
PROJ_PATH = os.path.join("output_ensemble", f"projections_ensemble_{SEASON}.csv")
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

EAST = {"Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
        "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers", "Miami Heat", "Milwaukee Bucks",
        "New York Knicks", "Orlando Magic", "Philadelphia 76ers", "Toronto Raptors", "Washington Wizards"}


def team_uncertainty_points():
    """
    How far off a preseason rating might be, in points per game. Backtest error (MAE) converts to a
    standard deviation of about 1.25 x MAE. Part of that spread comes from game-to-game luck, which
    the simulation adds on its own, so only the remainder is applied to the ratings.
    """
    total_sd_wins = BACKTEST_MAE * np.sqrt(np.pi / 2)
    luck_sd_wins = np.sqrt(82 * 0.25)
    rating_sd_wins = np.sqrt(max(total_sd_wins ** 2 - luck_sd_wins ** 2, 1.0))
    return rating_sd_wins / WINS_PER_POINT


# ---------------------------------------------------------------- inputs
def load_teams():
    if not os.path.exists(PROJ_PATH):
        raise SystemExit(f"Can't find {PROJ_PATH}. Run nba_wins_ensemble.py first.")
    proj = pd.read_csv(PROJ_PATH, encoding="utf-8-sig")
    proj_col = [c for c in proj.columns if c.startswith("tuned")][0]
    ts = pd.read_csv(os.path.join(DATA_DIR, "team_seasons.csv"))
    ids = ts[ts.season == ts.season.max()][["TEAM_ID", "TEAM_NAME"]]
    t = proj[["TEAM_NAME", proj_col]].rename(columns={proj_col: "preseason_proj"}).merge(ids, on="TEAM_NAME")
    if len(t) != 30:
        raise SystemExit(f"Expected 30 teams, matched {len(t)}. Check team names in {PROJ_PATH}.")
    t["conf"] = np.where(t.TEAM_NAME.isin(EAST), "East", "West")
    t["preseason_rating"] = (t.preseason_proj - 41) / WINS_PER_POINT
    return t.reset_index(drop=True)


BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
}


def schedule_from_cdn():
    """NBA.com's static schedule file (includes future games)."""
    import requests
    r = requests.get(SCHEDULE_URL, timeout=30, headers=BROWSER_HEADERS)
    r.raise_for_status()
    ls = r.json()["leagueSchedule"]
    if ls.get("seasonYear") != SEASON:
        raise ValueError(f"feed is for {ls.get('seasonYear')}, not {SEASON}")
    rows = []
    for day in ls["gameDates"]:
        for g in day["games"]:
            rows.append({"game_id": str(g["gameId"]), "date": g.get("gameDateEst") or day["gameDate"],
                         "home_id": g["homeTeam"]["teamId"], "away_id": g["awayTeam"]["teamId"]})
    return pd.DataFrame(rows)


def schedule_from_stats_api():
    """Same schedule through stats.nba.com (the nba_api package), which Version A already uses."""
    from nba_api.stats.endpoints import scheduleleaguev2
    df = scheduleleaguev2.ScheduleLeagueV2(season=SEASON, timeout=60).get_data_frames()[0]

    def pick(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        raise KeyError(f"none of {names} in schedule columns")

    return pd.DataFrame({
        "game_id": pick("gameId", "GAME_ID").astype(str),
        "date": pick("gameDateEst", "gameDate", "GAME_DATE"),
        "home_id": pick("homeTeam_teamId", "homeTeamId", "HOME_TEAM_ID"),
        "away_id": pick("awayTeam_teamId", "awayTeamId", "VISITOR_TEAM_ID"),
    })


def results_from_game_log():
    """Completed regular-season games with scores, from stats.nba.com's league game log."""
    from nba_api.stats.endpoints import leaguegamelog
    log = leaguegamelog.LeagueGameLog(season=SEASON, season_type_all_star="Regular Season",
                                      timeout=60).get_data_frames()[0]
    if log.empty:
        return pd.DataFrame(columns=["game_id", "home_pts", "away_pts"])
    log["GAME_ID"] = log.GAME_ID.astype(str)
    home = log[log.MATCHUP.str.contains(" vs. ")][["GAME_ID", "PTS"]].rename(columns={"PTS": "home_pts"})
    away = log[log.MATCHUP.str.contains(" @ ")][["GAME_ID", "PTS"]].rename(columns={"PTS": "away_pts"})
    return home.merge(away, on="GAME_ID").rename(columns={"GAME_ID": "game_id"})


def fetch_schedule(team_ids):
    """
    Regular-season schedule with results so far. Tries NBA.com's schedule file, then stats.nba.com,
    then the last saved copy, then a generic schedule.
    """
    path = os.path.join(DATA_DIR, f"schedule_{SEASON}.csv")
    sched = None
    for name, source in [("NBA.com schedule file", schedule_from_cdn),
                         ("stats.nba.com schedule", schedule_from_stats_api)]:
        try:
            sched = source()
            print(f"Schedule loaded from {name}.")
            break
        except Exception as e:
            print(f"Couldn't read the {name} ({e.__class__.__name__}: {str(e)[:80]}).")
    if sched is not None:
        sched["game_id"] = sched.game_id.str.zfill(10)
        sched = sched[sched.game_id.str.startswith("002")]  # 002 = regular season
        sched = sched[sched.home_id.isin(team_ids) & sched.away_id.isin(team_ids)].copy()
        sched["date"] = pd.to_datetime(sched.date, errors="coerce").dt.date
        try:
            res = results_from_game_log()
            res["game_id"] = res.game_id.str.zfill(10)
            sched = sched.merge(res, on="game_id", how="left")
        except Exception as e:
            print(f"Couldn't read completed results ({e.__class__.__name__}); treating all games as unplayed.")
            sched["home_pts"], sched["away_pts"] = None, None
        sched.to_csv(path, index=False)
        print(f"Schedule: {len(sched)} games, {sched.home_pts.notna().sum()} completed")
        return sched
    if os.path.exists(path):
        print("Using the last saved schedule.")
        return pd.read_csv(path, dtype={"game_id": str})
    print("Using a generic schedule: every team plays each opponent twice plus conference "
          "opponents twice more (86 games, scaled to 82).")
    return None


def fill_to_82(sched, teams, rng):
    """
    The NBA releases 1,200 games before the season and adds the last 30 in December, after the
    NBA Cup group stage. Until then, pair up teams that are short of 82 games (same conference
    first) with placeholder games so every team plays a full season.
    """
    counts = pd.concat([sched.home_id, sched.away_id]).value_counts().reindex(teams.TEAM_ID, fill_value=0)
    short = {tid: 82 - c for tid, c in counts.items() if c < 82}
    if not short:
        return sched
    conf = dict(zip(teams.TEAM_ID, teams.conf))
    added = []
    while sum(short.values()) >= 2:
        t = max(short, key=short.get)
        options = [o for o in short if o != t and short[o] > 0]
        if not options:
            break
        same = [o for o in options if conf[o] == conf[t]]
        o = rng.choice(same or options)
        home, away = (t, o) if rng.random() < 0.5 else (o, t)
        added.append({"game_id": f"tbd{len(added)}", "home_id": home, "away_id": away,
                      "home_pts": None, "away_pts": None})
        for x in (t, o):
            short[x] -= 1
            if short[x] == 0:
                del short[x]
    print(f"Added {len(added)} placeholder games so every team plays 82 "
          "(the NBA fills these in after the Cup group stage).")
    return pd.concat([sched, pd.DataFrame(added)], ignore_index=True)


def generic_schedule(teams):
    rows = []
    for i in range(30):
        for j in range(30):
            if i == j:
                continue
            reps = 2 if teams.conf[i] == teams.conf[j] else 1
            for _ in range(reps):
                rows.append({"home_id": teams.TEAM_ID[i], "away_id": teams.TEAM_ID[j],
                             "home_pts": None, "away_pts": None})
    return pd.DataFrame(rows)


# ------------------------------------------------------- in-season update
def update_ratings(teams, played):
    """Blend each preseason rating with actual point differential so far (adjusted for home court)."""
    t = teams.copy()
    if played.empty:
        t["GP"], t["W"], t["L"], t["margin"] = 0, 0, 0, 0.0
        t["rating"] = t.preseason_rating
        return t
    diff = played.home_pts.astype(float) - played.away_pts.astype(float)
    long = pd.concat([
        pd.DataFrame({"TEAM_ID": played.home_id, "won": diff > 0, "adj_margin": diff - HOME_COURT}),
        pd.DataFrame({"TEAM_ID": played.away_id, "won": diff < 0, "adj_margin": -diff + HOME_COURT}),
    ])
    agg = long.groupby("TEAM_ID").agg(GP=("won", "size"), W=("won", "sum"), margin=("adj_margin", "mean"))
    t = t.merge(agg, left_on="TEAM_ID", right_index=True, how="left")
    t[["GP", "W", "margin"]] = t[["GP", "W", "margin"]].fillna(0)
    t["GP"], t["W"] = t.GP.astype(int), t.W.astype(int)
    t["L"] = t.GP - t.W
    t["rating"] = (PRIOR_GAMES * t.preseason_rating + t.GP * t.margin) / (PRIOR_GAMES + t.GP)
    return t


# ------------------------------------------------------------ simulation
def p_home_win(r_home, r_away):
    return norm.cdf((r_home - r_away + HOME_COURT) / GAME_SD)


def simulate(teams, remaining, scale=1.0, rng=None):
    n = len(teams)
    idx = {tid: i for i, tid in enumerate(teams.TEAM_ID)}
    h = remaining.home_id.map(idx).to_numpy()
    a = remaining.away_id.map(idx).to_numpy()

    # uncertainty about true strength shrinks as games are played
    sd = team_uncertainty_points() * np.sqrt(PRIOR_GAMES / (PRIOR_GAMES + teams.GP.to_numpy()))
    ratings = teams.rating.to_numpy() + rng.normal(0, 1, (N_SIMS, n)) * sd

    wins = np.tile(teams.W.to_numpy().astype(float), (N_SIMS, 1))
    if len(remaining):
        home_won = rng.random((N_SIMS, len(h))) < p_home_win(ratings[:, h], ratings[:, a])
        onehot_h = np.eye(n)[h]
        onehot_a = np.eye(n)[a]
        wins += (home_won @ onehot_h + (~home_won) @ onehot_a) * scale
    return ratings, wins


def best_of_seven(r_high, r_low, rng):
    """Higher seed hosts games 1, 2, 5 and 7. Returns True if the higher seed wins the series."""
    home = np.array([1, 1, 0, 0, 1, 0, 1], bool)
    p = np.where(home, p_home_win(r_high, r_low), 1 - p_home_win(r_low, r_high))
    return (rng.random(7) < p).sum() >= 4


def postseason(teams, ratings, wins, rng):
    n = len(teams)
    counts = {k: np.zeros(n) for k in ["top6", "playin", "playoffs", "round2", "conf_finals", "finals", "title"]}
    seeds_hist = np.zeros((n, 16))
    conf_idx = {c: np.where(teams.conf == c)[0] for c in ["East", "West"]}
    for s in range(N_SIMS):
        r = ratings[s]
        champs = []
        for c, members in conf_idx.items():
            order = members[np.lexsort((rng.random(len(members)), -wins[s, members]))]  # random tiebreaks
            for seed, t in enumerate(order, 1):
                seeds_hist[t, seed] += 1
            counts["top6"][order[:6]] += 1
            counts["playin"][order[6:10]] += 1
            s7, s8, s9, s10 = order[6:10]
            win78 = s7 if rng.random() < p_home_win(r[s7], r[s8]) else s8
            lose78 = s8 if win78 == s7 else s7
            win910 = s9 if rng.random() < p_home_win(r[s9], r[s10]) else s10
            eighth = lose78 if rng.random() < p_home_win(r[lose78], r[win910]) else win910
            field = list(order[:6]) + [win78, eighth]
            counts["playoffs"][field] += 1
            bracket = [field[0], field[7], field[3], field[4], field[2], field[5], field[1], field[6]]
            for rnd in ["round2", "conf_finals", "finals"]:
                nxt = []
                for x, y in zip(bracket[::2], bracket[1::2]):
                    hi, lo = (x, y) if wins[s, x] >= wins[s, y] else (y, x)
                    nxt.append(hi if best_of_seven(r[hi], r[lo], rng) else lo)
                bracket = nxt
                counts[rnd][bracket] += 1
            champs.append(bracket[0])
        x, y = champs
        hi, lo = (x, y) if wins[s, x] >= wins[s, y] else (y, x)
        counts["title"][hi if best_of_seven(r[hi], r[lo], rng) else lo] += 1
    return {k: v / N_SIMS * 100 for k, v in counts.items()}, seeds_hist / N_SIMS * 100


# ---------------------------------------------------------------- output
def chart(summary, path, label):
    s = summary.sort_values("mean_wins")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.hlines(y, s.p10_wins, s.p90_wins, color="#9db8d6", lw=6, label="80% range")
    ax.scatter(s.mean_wins, y, color="#1f3b5c", zorder=3, s=22, label="Average")
    if s.GP.max() > 0:
        ax.scatter(s.W, y, color="#c0504d", marker="|", s=90, zorder=3, label="Wins so far")
    ax.set_yticks(y, [f"{t}  ({p:.0f}%)" for t, p in zip(s.TEAM_NAME, s.playoffs_pct)], fontsize=8)
    ax.axvline(41, color="#888", lw=1, ls="--")
    ax.set_xlabel("Wins")
    ax.set_title(f"{SEASON} simulated wins, {label}\n(playoff odds in parentheses, {N_SIMS:,} simulations)")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    teams = load_teams()

    sched = fetch_schedule(set(teams.TEAM_ID))
    scale = 1.0
    if sched is None:
        sched = generic_schedule(teams)
        scale = 82 / 86
    else:
        sched = fill_to_82(sched, teams, rng)
    played = sched[sched.home_pts.notna()]
    remaining = sched[sched.home_pts.isna()]

    teams = update_ratings(teams, played)
    ratings, wins = simulate(teams, remaining, scale, rng)
    odds, seeds = postseason(teams, ratings, wins, rng)

    today = dt.date.today().isoformat()
    label = "preseason" if played.empty else f"as of {today} ({len(played)} games played)"
    summary = teams[["TEAM_NAME", "conf", "preseason_proj", "GP", "W", "L", "rating"]].copy()
    summary["pace_82"] = np.where(summary.GP > 0, summary.W / summary.GP.clip(lower=1) * 82, np.nan)
    summary["mean_wins"] = wins.mean(axis=0)
    summary["p10_wins"] = np.percentile(wins, 10, axis=0)
    summary["p90_wins"] = np.percentile(wins, 90, axis=0)
    for k, v in odds.items():
        summary[f"{k}_pct"] = v
    summary["most_likely_seed"] = seeds[:, 1:].argmax(axis=1) + 1
    summary = summary.sort_values(["conf", "mean_wins"], ascending=[True, False]).round(1)

    summary.to_csv(os.path.join(OUT_DIR, "sim_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"\n{SEASON} simulation, {label}:")
    cols = ["TEAM_NAME", "W", "L", "mean_wins", "p10_wins", "p90_wins", "playoffs_pct", "finals_pct", "title_pct"]
    for c in ["East", "West"]:
        print(f"\n{c}")
        print(summary[summary.conf == c][cols].to_string(index=False))

    # win distribution for the dashboard's histogram
    dist = []
    for i, name in enumerate(teams.TEAM_NAME):
        vals, cnt = np.unique(np.round(wins[:, i]).astype(int), return_counts=True)
        dist += [{"TEAM_NAME": name, "wins": v, "pct": c / N_SIMS * 100} for v, c in zip(vals, cnt)]
    pd.DataFrame(dist).to_csv(os.path.join(OUT_DIR, "win_distribution.csv"), index=False, encoding="utf-8-sig")

    # dated snapshot for tracking over the season (re-running on the same day replaces that day's row)
    hist_path = os.path.join(OUT_DIR, "tracker_history.csv")
    snap = summary.assign(snapshot_date=today)[["snapshot_date", "TEAM_NAME", "conf", "GP", "W", "L", "pace_82",
                                                "preseason_proj", "mean_wins", "p10_wins", "p90_wins",
                                                "playoffs_pct", "title_pct", "rating"]]
    if os.path.exists(hist_path):
        old = pd.read_csv(hist_path, encoding="utf-8-sig")
        snap = pd.concat([old[old.snapshot_date != today], snap], ignore_index=True)
    snap.to_csv(hist_path, index=False, encoding="utf-8-sig")

    chart(summary, os.path.join(OUT_DIR, "sim_ranges.png"), label)
    print(f"\nSaved results to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
