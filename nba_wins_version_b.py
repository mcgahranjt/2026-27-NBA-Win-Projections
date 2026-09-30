"""
NBA win projection - Version B (player-level model)

Builds each team's rating up from its players instead of from last season's team numbers.

  1. Player ratings: Box Plus/Minus (BPM) from Basketball-Reference, 2007-08 onward
  2. Projected rating for each player = weighted average of his last three seasons
     (60/30/10 across the seasons he played, also weighted by minutes), pulled toward a
     below-average prior when the sample is small, plus an aging adjustment estimated from the data.
     How hard to pull is tuned on 2015-2020, before the test seasons.
  3. Projected minutes = minutes per game from his most recent full season x 68 games,
     then filled by depth chart until the team reaches 19,680 minutes (48 x 5 x 82)
  4. Team rating = minutes-weighted average player rating x 5 (roughly points per 100 possessions)
  5. Wins = linear fit of actual wins on team rating, trained on past seasons
  6. A "blend" model also adds last season's team net rating and the script keeps whichever
     backtests better

Requires the ./data folder from Version A (team_seasons.csv, player_seasons.csv).
Run nba_wins_version_a.py first if you haven't.

Usage:
  pip install nba_api pandas scikit-learn matplotlib requests lxml
  python3 nba_wins_version_b.py
"""

import os
import re
import time
import unicodedata
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_DIR = "data"
OUT_DIR = "output_b"

BR_FIRST_END_YEAR = 2008     # Basketball-Reference labels seasons by end year (2008 = 2007-08)
LAST_COMPLETED = 2025        # 2025-26, labelled by start year like Version A
TARGET = LAST_COMPLETED + 1  # 2026-27
FIRST_TARGET = 2010          # first season with three prior seasons of player data
TEST_SEASONS = range(LAST_COMPLETED - 4, LAST_COMPLETED + 1)
GAMES = 82
TEAM_MINUTES = 48 * 5 * GAMES

# projection settings
SEASON_WEIGHTS = [0.6, 0.3, 0.1]  # last season, two seasons ago, three seasons ago
PRIOR_BPM = -2.0                  # where small samples get pulled toward (a fringe rotation player)
PRIOR_MINUTE_OPTIONS = [250, 500, 750, 1000, 1500]  # how hard small samples get pulled toward the prior;
                                              # the script picks the best one on 2015-2020 (before the test seasons)
TUNE_SEASONS = range(2015, 2021)
MAX_MINUTES = 36 * 70             # one player's projected minutes are capped at ~36 a game
ROOKIE_BPM = -2.5
ROOKIE_MINUTES = 600
EXPECTED_GAMES = 68
MIN_GAMES_FOR_MPG = 20

# Basketball-Reference team codes -> NBA.com team IDs (IDs stay with the franchise through moves)
BR_TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "CLE": 1610612739, "NOH": 1610612740, "NOP": 1610612740,
    "NOK": 1610612740, "CHI": 1610612741, "DAL": 1610612742, "DEN": 1610612743, "GSW": 1610612744,
    "HOU": 1610612745, "LAC": 1610612746, "LAL": 1610612747, "MIA": 1610612748, "MIL": 1610612749,
    "MIN": 1610612750, "NJN": 1610612751, "BRK": 1610612751, "NYK": 1610612752, "ORL": 1610612753,
    "IND": 1610612754, "PHI": 1610612755, "PHO": 1610612756, "POR": 1610612757, "SAC": 1610612758,
    "SAS": 1610612759, "OKC": 1610612760, "SEA": 1610612760, "TOR": 1610612761, "UTA": 1610612762,
    "MEM": 1610612763, "WAS": 1610612764, "DET": 1610612765, "CHA": 1610612766, "CHO": 1610612766,
}

# If the script reports roster players it couldn't match, add them here:
# "name as NBA.com spells it (lowercase, no punctuation)": "name as Basketball-Reference spells it"
NAME_FIXES = {
    "ronald holland": "ron holland",
}


def season_str(start_year):
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fix_mojibake(text):
    """Repair names like 'JokiÄ\x87' that were decoded with the wrong character set."""
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = text.encode(enc).decode("utf-8")
            if fixed != text:
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def name_key(name):
    """Normalize a player name so NBA.com and Basketball-Reference spellings line up."""
    n = unicodedata.normalize("NFKD", fix_mojibake(str(name)))
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    n = re.sub(r"[.'`,*]", "", n).replace("-", " ")
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return NAME_FIXES.get(n, n)


# ---------------------------------------------------------------- 1. data
def fetch_br_season(end_year):
    """One season of Basketball-Reference advanced stats, cached to CSV."""
    path = os.path.join(DATA_DIR, f"br_advanced_{end_year}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    import requests

    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_advanced.html"
    print(f"Basketball-Reference {end_year - 1}-{str(end_year)[-2:]}")
    for attempt in range(4):
        time.sleep(4)  # stay well under their 20-requests-per-minute limit
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (personal research project)"}, timeout=30)
        if r.status_code == 200:
            break
        print(f"  status {r.status_code}, retrying in 60s")
        time.sleep(60)
    r.raise_for_status()
    r.encoding = "utf-8"  # Basketball-Reference pages are UTF-8; without this, accented names get garbled

    tables = pd.read_html(StringIO(r.text))
    df = next(t for t in tables if "BPM" in [c[-1] if isinstance(c, tuple) else c for c in t.columns])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df = df.rename(columns={"Tm": "Team"})
    df = df[(df["Player"] != "Player") & (df["Player"] != "League Average")].copy()
    df = df[["Player", "Age", "Team", "G", "MP", "BPM"]].dropna(subset=["Player", "Team"])
    for c in ["Age", "G", "MP", "BPM"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Player"] = df["Player"].str.replace("*", "", regex=False).str.strip()
    df["row"] = range(len(df))
    df.to_csv(path, index=False)
    return df


def is_total_row(team):
    return team == "TOT" or bool(re.fullmatch(r"\dTM", str(team)))


def player_seasons_br():
    """One row per player-season: season totals plus the team he started the season with."""
    rows = []
    for end_year in range(BR_FIRST_END_YEAR, LAST_COMPLETED + 2):
        df = fetch_br_season(end_year).sort_values("row")
        for player, g in df.groupby("Player", sort=False):
            totals = g[g["Team"].map(is_total_row)]
            teams = g[~g["Team"].map(is_total_row)]
            if teams.empty:
                continue
            tot = totals.iloc[0] if not totals.empty else teams.iloc[0]
            rows.append({
                "key": name_key(player), "player": player, "season": end_year - 1,
                "age": tot["Age"], "G": tot["G"], "MP": tot["MP"], "BPM": tot["BPM"],
                "first_team": teams.iloc[0]["Team"],
            })
    out = pd.DataFrame(rows).drop_duplicates(["key", "season"])
    out["TEAM_ID"] = out["first_team"].map(BR_TEAM_IDS)
    missing = out.loc[out["TEAM_ID"].isna(), "first_team"].unique()
    if len(missing):
        print(f"Warning: unmapped team codes {missing}")
    return out


def fetch_target_rosters_named(team_ids):
    """2026-27 rosters with names, ages and experience from NBA.com."""
    path = os.path.join(DATA_DIR, f"rosters_named_{TARGET}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from nba_api.stats.endpoints import commonteamroster

    rows = []
    for tid in team_ids:
        for attempt in range(4):
            try:
                time.sleep(0.8)
                df = commonteamroster.CommonTeamRoster(team_id=int(tid), season=season_str(TARGET),
                                                       timeout=60).get_data_frames()[0]
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Couldn't get the roster for team {tid} from NBA.com - try again in a few minutes.")
        rows.append(pd.DataFrame({"TEAM_ID": tid, "PLAYER_ID": df["PLAYER_ID"], "player": df["PLAYER"],
                                  "age_now": pd.to_numeric(df["AGE"], errors="coerce"), "EXP": df["EXP"]}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


# ----------------------------------------------------------- 2. projections
def aging_curve(ps, last_season):
    """Average one-year change in BPM by age, using only seasons up to last_season."""
    a = ps[ps.season <= last_season - 1][["key", "season", "age", "BPM", "MP"]]
    b = ps[ps.season <= last_season][["key", "season", "BPM", "MP"]].copy()
    b["season"] -= 1
    pairs = a.merge(b, on=["key", "season"], suffixes=("", "_next"))
    pairs = pairs[(pairs.MP >= 500) & (pairs.MP_next >= 500)]
    pairs["w"] = 2 / (1 / pairs.MP + 1 / pairs.MP_next)  # harmonic mean of the two seasons' minutes
    pairs["delta"] = pairs.BPM_next - pairs.BPM
    curve = pairs.groupby("age").apply(lambda g: np.average(g.delta, weights=g.w), include_groups=False)
    ages = np.arange(18, 42)
    curve = curve.reindex(ages).interpolate(limit_direction="both")
    return curve.rolling(3, center=True, min_periods=1).mean()  # smooth out noisy single ages


def project_players(ps, target, curve, prior_minutes):
    """Projected BPM and raw minutes for every player with history, going into `target`."""
    hist = ps[(ps.season >= target - 3) & (ps.season <= target - 1)].copy()
    hist["sw"] = hist.season.map({target - 1: SEASON_WEIGHTS[0], target - 2: SEASON_WEIGHTS[1],
                                  target - 3: SEASON_WEIGHTS[2]})
    # re-weight across the seasons a player actually played, so a season missed to injury
    # doesn't make his earlier seasons count for less
    hist["w"] = hist.sw / hist.groupby("key").sw.transform("sum") * hist.MP
    agg = hist.groupby("key").agg(
        num=("BPM", lambda s: (s * hist.loc[s.index, "w"]).sum()),
        den=("w", "sum"),
        last_age=("age", "last"),
        last_season=("season", "max"),
    )
    agg["bpm_raw"] = (agg.num + prior_minutes * PRIOR_BPM) / (agg.den + prior_minutes)
    agg["age_next"] = agg.last_age + (target - agg.last_season)
    agg["aging"] = agg.last_age.round().clip(18, 41).map(curve).fillna(0)
    agg["proj_bpm"] = agg.bpm_raw + agg.aging

    # minutes: most recent season (of the last two) with enough games to trust his minutes per game
    recent = hist[(hist.season >= target - 2) & (hist.G >= MIN_GAMES_FOR_MPG)].sort_values("season")
    mpg = recent.groupby("key").apply(lambda g: g.MP.iloc[-1] / g.G.iloc[-1], include_groups=False)
    agg["raw_min"] = (mpg * EXPECTED_GAMES).reindex(agg.index).fillna(300).clip(upper=MAX_MINUTES)
    return agg[["proj_bpm", "raw_min", "age_next"]]


def allocate_minutes(raw):
    """
    Fill a team's 19,680 minutes by depth chart: players with the biggest projected roles get
    their minutes first, and whoever is left at the end of the bench gets what remains.
    If the roster doesn't add up to a full season, everyone is scaled up (capped at MAX_MINUTES).
    """
    order = raw.astype(float).sort_values(ascending=False)
    total = order.sum()
    if total >= TEAM_MINUTES:
        before = order.cumsum() - order
        alloc = (TEAM_MINUTES - before).clip(lower=0).combine(order, min)
    else:
        alloc = order.copy()
        for _ in range(10):
            short = TEAM_MINUTES - alloc.sum()
            room = alloc < MAX_MINUTES
            if short < 1 or not room.any():
                break
            alloc[room] = (alloc[room] * (1 + short / alloc[room].sum())).clip(upper=MAX_MINUTES)
    return alloc.reindex(raw.index)


def team_ratings(roster, proj):
    """roster: [key, TEAM_ID]. Players with no history are treated as rookies."""
    r = roster.merge(proj, left_on="key", right_index=True, how="left")
    r["rookie"] = r.proj_bpm.isna()
    r["proj_bpm"] = r.proj_bpm.fillna(ROOKIE_BPM)
    r["raw_min"] = r.raw_min.fillna(ROOKIE_MINUTES)
    r["proj_min"] = r.groupby("TEAM_ID").raw_min.transform(allocate_minutes)
    r["contrib"] = r.proj_bpm * r.proj_min
    team = (r.groupby("TEAM_ID").contrib.sum() / TEAM_MINUTES * 5).rename("team_rating")
    return team, r


def build_history(ps, teams, prior_minutes, curves):
    rows = []
    for target in range(FIRST_TARGET, LAST_COMPLETED + 1):
        proj = project_players(ps, target, curves[target], prior_minutes)
        roster = ps.loc[ps.season == target, ["key", "TEAM_ID"]].dropna()
        team, _ = team_ratings(roster, proj)
        t = teams[teams.season == target][["TEAM_ID", "TEAM_NAME", "W_PCT"]].set_index("TEAM_ID")
        prev = teams[teams.season == target - 1].set_index("TEAM_ID")
        df = t.join(team).join(prev[["NET_RATING", "W_PCT"]].rename(
            columns={"NET_RATING": "prev_net_rating", "W_PCT": "prev_w_pct"}))
        df["season"] = target
        rows.append(df.reset_index())
    h = pd.concat(rows, ignore_index=True)
    h["wins_82"] = h.W_PCT * GAMES
    h["prev_wins_82"] = h.prev_w_pct * GAMES
    return h


def tune_prior_minutes(ps, teams, curves):
    """Pick how strongly to pull ratings toward average, using seasons before the test period."""
    tune = {}
    for pm in PRIOR_MINUTE_OPTIONS:
        h = build_history(ps, teams, pm, curves)
        errs = []
        for s in TUNE_SEASONS:
            tr = h[(h.season < s) & h.prev_net_rating.notna()]
            te = h[h.season == s]
            m = LinearRegression().fit(tr[["team_rating"]], tr.wins_82)
            errs.append(mean_absolute_error(te.wins_82, normalize_to_league(m.predict(te[["team_rating"]]))))
        tune[pm] = np.mean(errs)
    return min(tune, key=tune.get), tune


# ------------------------------------------------------------ 3. evaluate
MODELS = {"B": ["team_rating"], "blend": ["team_rating", "prev_net_rating"]}


def normalize_to_league(pred):
    return pred + (GAMES * 15 - pred.sum()) / len(pred)


def backtest(hist):
    out = []
    for s in TEST_SEASONS:
        train = hist[(hist.season < s) & hist.prev_net_rating.notna()]
        test = hist[hist.season == s]
        row = {"season": season_str(s)}
        for name, feats in MODELS.items():
            m = LinearRegression().fit(train[feats], train.wins_82)
            row[f"model_{name}_MAE"] = mean_absolute_error(test.wins_82, normalize_to_league(m.predict(test[feats])))
        row["baseline_last_year_MAE"] = mean_absolute_error(test.wins_82, test.prev_wins_82)
        row["baseline_41_MAE"] = mean_absolute_error(test.wins_82, np.full(len(test), 41.0))
        out.append(row)
    bt = pd.DataFrame(out)
    a_path = os.path.join("output", "backtest.csv")  # Version A results, if available
    if os.path.exists(a_path):
        a = pd.read_csv(a_path)
        if "model_avg_age_MAE" in a:
            bt = bt.merge(a[["season", "model_avg_age_MAE"]].rename(
                columns={"model_avg_age_MAE": "version_A_MAE"}), on="season", how="left")
    return bt


def chart(proj, path, label):
    proj = proj.sort_values("projected_wins")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(proj.TEAM_NAME, proj.projected_wins, color="#3b6ea5")
    ax.axvline(41, color="#888", lw=1, ls="--")
    for i, v in enumerate(proj.projected_wins):
        ax.text(v + 0.4, i, f"{v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("Projected wins")
    ax.set_title(f"{season_str(TARGET)} projected wins (Version B, {label})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    teams_path = os.path.join(DATA_DIR, "team_seasons.csv")
    if not os.path.exists(teams_path):
        raise SystemExit("Run nba_wins_version_a.py first - Version B reuses its data folder.")
    teams = pd.read_csv(teams_path)

    ps = player_seasons_br()
    curves = {t: aging_curve(ps, t - 1) for t in range(FIRST_TARGET, TARGET + 1)}

    prior_minutes, tune = tune_prior_minutes(ps, teams, curves)
    print("Tuning (average error on 2015-16 to 2020-21): "
          + ", ".join(f"{k} min = {v:.2f}" for k, v in tune.items()) + f" -> using {prior_minutes}")
    hist = build_history(ps, teams, prior_minutes, curves)
    hist.to_csv(os.path.join(OUT_DIR, "training_data_b.csv"), index=False)

    bt = backtest(hist)
    print("\nBacktest (mean absolute error in wins, lower is better):")
    print(bt.round(2).to_string(index=False))
    means = bt.drop(columns="season").mean()
    print("\nAverage:")
    print(means.round(2).to_string())
    bt.to_csv(os.path.join(OUT_DIR, "backtest_b.csv"), index=False)

    best = min(MODELS, key=lambda n: means[f"model_{n}_MAE"])
    train = hist[hist.prev_net_rating.notna()]
    for name, feats in MODELS.items():
        m = LinearRegression().fit(train[feats], train.wins_82)
        print(f"Model {name}: wins = {m.intercept_:.1f}"
              + "".join(f" + {c:.2f} x {f}" for c, f in zip(m.coef_, feats)))
    print(f"Using '{best}' for projections.")
    model = LinearRegression().fit(train[MODELS[best]], train.wins_82)

    # 2026-27
    team_ids = teams.loc[teams.season == LAST_COMPLETED, "TEAM_ID"].unique()
    rosters = fetch_target_rosters_named(team_ids)
    rosters["key"] = rosters.player.map(name_key)
    known = set(ps.key)
    unmatched = rosters[(~rosters.key.isin(known)) & (rosters.EXP.astype(str) != "R")]
    if len(unmatched):
        print(f"\n{len(unmatched)} non-rookie roster players not found on Basketball-Reference "
              "(treated as rookies). Add them to NAME_FIXES if they matter:")
        print(", ".join(unmatched.player))

    proj = project_players(ps, TARGET, curves[TARGET], prior_minutes)
    team, players = team_ratings(rosters[["key", "player", "TEAM_ID"]], proj)

    last = teams[teams.season == LAST_COMPLETED].set_index("TEAM_ID")
    out = last[["TEAM_NAME"]].join(team)
    out["prev_net_rating"] = last.NET_RATING
    out["prev_wins_82"] = last.W_PCT * GAMES
    out["projected_wins"] = normalize_to_league(model.predict(out[MODELS[best]])).round(1)
    out = out.sort_values("projected_wins", ascending=False)
    cols = ["TEAM_NAME", "projected_wins", "prev_wins_82", "team_rating", "prev_net_rating"]
    out[cols].to_csv(os.path.join(OUT_DIR, f"projections_b_{season_str(TARGET)}.csv"), index=False,
                     encoding="utf-8-sig")
    print(f"\n{season_str(TARGET)} projections:")
    print(out[cols].round(1).to_string(index=False))

    players = players.merge(last[["TEAM_NAME"]], left_on="TEAM_ID", right_index=True)
    players = players.sort_values(["TEAM_NAME", "proj_min"], ascending=[True, False])
    players["proj_mpg"] = players.proj_min / EXPECTED_GAMES
    players[["TEAM_NAME", "player", "proj_bpm", "proj_min", "proj_mpg", "rookie"]].round(1).to_csv(
        os.path.join(OUT_DIR, f"player_projections_{season_str(TARGET)}.csv"), index=False,
        encoding="utf-8-sig")  # utf-8-sig so Excel shows accented names correctly
    chart(out, os.path.join(OUT_DIR, f"projections_b_{season_str(TARGET)}.png"), best)
    print(f"\nSaved results to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
