"""
NBA power rankings

Rates every team in points per game versus an average team on a neutral court, and ranks them.
Run it any time during the season to update the rankings.

How the ratings work:
  - Before the season: the weighted-statistics preseason power points from
    nba_preseason_power.py (falls back to (combined projected wins - 41) / 2.7 if that hasn't been run)
  - Once games are played: every game's scoring is fit with a ridge regression (below), with each
    team pulled toward its preseason value. The preseason rating counts about as much as 20 games,
    so it carries early in the season and results take over later.
  - Recent games count more than old ones (60-day half-life), and per-game efficiencies more than
    25 points per 100 from average are capped so garbage time doesn't distort the ratings.
  - The power rating (points better than average per game) = net rating at league-average pace.

Offense, defense and pace:
  - The single rating is split into offense (points scored per 100 possessions vs. average) and
    defense (points allowed per 100 vs. average, positive = better), each adjusted for opponents.
    Net rating = offense + defense. Both are fit together from every game's points and possessions,
    with the same preseason anchor, recency weighting and back-to-back adjustment.
  - Pace (possessions per game) is fit the same way: game pace = league pace + both teams' pace
    tendencies. Pace settles faster than efficiency, so its preseason anchor is lighter.
  - Possessions per game are estimated from the box score: FGA + 0.44 x FTA - OREB + TOV,
    averaged across both teams.
  - Preseason anchors: net = the preseason power points; offense/defense split it using last
    season's offense/defense balance (half carried over); pace = last season's pace, 60% carried over.

Each run saves a dated snapshot to output_power/power_history.csv, so rank movement can be
tracked over the season (and charted in Tableau).

Outputs: output_power/power_rankings.csv, team_ratings.csv (offense, defense, net, pace and ranks),
league_levels.csv, power_history.csv, and game_projections.csv (projected margin, total and win
probability for the next two weeks of games; also appended to game_projection_history.csv for grading). nba_season_sim.py reads team_ratings.csv when it was written
the same day, so the rankings, workbook and simulation all use one set of ratings.

If NBA_Spread_Calculator.xlsx is in this folder, each run also writes the new rankings into its
Power Rankings sheet and replaces its Team Ratings sheet. The first run adds a Matchup Projection
sheet (projected possessions, scores, total, line and win probability); after that it's never
touched, and no other sheet is changed. Close the workbook before running.

Needs output_ensemble/projections_ensemble_2026-27.csv, data/team_seasons.csv and
nba_season_sim.py in the same folder (it reuses the simulator's schedule and results code).

Usage:
  python3 nba_power_ratings.py
"""

import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_season_sim as S

OUT_DIR = "output_power"
HOME_COURT = S.HOME_COURT   # 2.3 points
GAME_SD = S.GAME_SD         # 12 points
PRIOR_GAMES = 20            # the preseason rating counts about as much as this many games
HALF_LIFE_DAYS = 60         # a game from 60 days ago counts half as much as one from today
MARGIN_CAP = 20             # blowouts beyond 20 points count as 20
B2B_PENALTY = 1.8           # points lost on the second night of a back-to-back
PACE_PRIOR_GAMES = 10       # pace settles quickly, so the preseason pace counts as ~10 games
PACE_CARRYOVER = 0.6        # share of last season's pace difference expected to carry over
SPLIT_CARRYOVER = 0.5       # share of last season's offense/defense balance carried into the split
PPP_CAP = 25                # per-100 game efficiencies more than 25 from average are capped
DEFAULT_PACE = 99.0         # used only if no pace data can be found
DEFAULT_ORTG = 114.0
PRESEASON_PATH = os.path.join("output_preseason_power", f"preseason_power_{S.SEASON}.csv")
CALCULATOR_PATH = "NBA_Spread_Calculator.xlsx"


def back_to_back_adjustment(sched):
    """Points to add to each game's expected home margin for back-to-backs (positive favors home)."""
    s = sched.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    dated = s[s.date.notna()]
    long = pd.concat([dated[["game_id", "date"]].assign(team=dated.home_id, side="home"),
                      dated[["game_id", "date"]].assign(team=dated.away_id, side="away")])
    long = long.sort_values(["team", "date"])
    long["b2b"] = long.groupby("team").date.diff().dt.days == 1
    flags = long.pivot_table(index="game_id", columns="side", values="b2b", aggfunc="first")
    home_b2b = s.game_id.map(flags.get("home", pd.Series(dtype=bool))).fillna(False).astype(int)
    away_b2b = s.game_id.map(flags.get("away", pd.Series(dtype=bool))).fillna(False).astype(int)
    s["rest_adj"] = B2B_PENALTY * (away_b2b - home_b2b)
    return s


def fetch_last_season_advanced():
    """Last season's offensive rating, defensive rating and pace by team (NBA.com), cached."""
    path = os.path.join(S.DATA_DIR, f"team_advanced_{S.SEASON[:4]}_prev.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        prev = f"{int(S.SEASON[:4]) - 1}-{S.SEASON[2:4]}"
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=prev, measure_type_detailed_defense="Advanced", timeout=60).get_data_frames()[0]
        df = df[["TEAM_ID", "OFF_RATING", "DEF_RATING", "PACE"]]
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        print(f"Couldn't get last season's offense/defense/pace ({e.__class__.__name__}); using even splits.")
        return None


def fetch_box_log():
    """This season's team box scores (points and possession inputs) by game, cached as a fallback."""
    path = os.path.join(S.DATA_DIR, f"team_box_log_{S.SEASON}.csv")
    try:
        from nba_api.stats.endpoints import leaguegamelog
        log = leaguegamelog.LeagueGameLog(season=S.SEASON, season_type_all_star="Regular Season",
                                          timeout=60).get_data_frames()[0]
        if log.empty:
            return pd.DataFrame(columns=["game_id", "poss"])
        log = log[["GAME_ID", "TEAM_ID", "MATCHUP", "PTS", "FGA", "FTA", "OREB", "TOV"]]
        log.to_csv(path, index=False)
    except Exception as e:
        if not os.path.exists(path):
            print(f"Couldn't get box scores ({e.__class__.__name__}); using league-average possessions.")
            return pd.DataFrame(columns=["game_id", "poss"])
        log = pd.read_csv(path)
    log["GAME_ID"] = log.GAME_ID.astype(str).str.zfill(10)
    log["est"] = log.FGA + 0.44 * log.FTA - log.OREB + log.TOV
    poss = log.groupby("GAME_ID").est.mean().rename("poss").reset_index().rename(columns={"GAME_ID": "game_id"})
    return poss


def add_priors(teams, adv):
    """Preseason anchors for offense, defense and pace."""
    t = teams.copy()
    if adv is not None and len(adv):
        a = adv.set_index("TEAM_ID")
        lg_o, lg_p = a.OFF_RATING.mean(), a.PACE.mean()
        off_last = t.TEAM_ID.map(a.OFF_RATING) - lg_o
        def_last = lg_o - t.TEAM_ID.map(a.DEF_RATING)     # positive = better defense
        balance = (off_last - def_last).fillna(0)
        pace_dev = (t.TEAM_ID.map(a.PACE) - lg_p).fillna(0) * PACE_CARRYOVER
    else:
        lg_o, lg_p = DEFAULT_ORTG, DEFAULT_PACE
        balance = pace_dev = pd.Series(0.0, index=t.index)
    net = t.preseason_rating * 100 / lg_p          # points per game -> per 100 possessions
    t["off_prior"] = net / 2 + SPLIT_CARRYOVER * balance / 2
    t["def_prior"] = net / 2 - SPLIT_CARRYOVER * balance / 2
    t["pace_prior"] = pace_dev
    return t, lg_o, lg_p


def recency_weights(dates, as_of):
    days_ago = (pd.Timestamp(as_of) - pd.to_datetime(dates)).dt.days.clip(lower=0).fillna(0)
    return 0.5 ** (days_ago.to_numpy() / HALF_LIFE_DAYS)


def fit_off_def(teams, games, lg_ortg, as_of):
    """
    Ridge regression on both teams' scoring efficiency in every game, anchored to the preseason split:
        100 x home pts / poss - league = home offense - away defense + home court/2 + rest/2
        100 x away pts / poss - league = away offense - home defense - home court/2 - rest/2
    """
    n = len(teams)
    prior = np.concatenate([teams.off_prior.to_numpy(), teams.def_prior.to_numpy()])
    if games.empty:
        return prior[:n].copy(), prior[n:].copy()
    idx = {tid: i for i, tid in enumerate(teams.TEAM_ID)}
    h = games.home_id.map(idx).to_numpy()
    a = games.away_id.map(idx).to_numpy()
    P = games.poss.to_numpy()
    hc = HOME_COURT * 100 / P / 2          # home court and rest, converted to per-100 and split
    rest = games.rest_adj.to_numpy() * 100 / P / 2
    y_home = (100 * games.home_pts.to_numpy() / P - lg_ortg).clip(-PPP_CAP, PPP_CAP) - hc - rest
    y_away = (100 * games.away_pts.to_numpy() / P - lg_ortg).clip(-PPP_CAP, PPP_CAP) + hc + rest
    m = len(games)
    X = np.zeros((2 * m, 2 * n))
    X[np.arange(m), h] = 1          # home offense
    X[np.arange(m), n + a] = -1     # away defense
    X[m + np.arange(m), a] = 1      # away offense
    X[m + np.arange(m), n + h] = -1 # home defense
    y = np.concatenate([y_home, y_away]) - X @ prior
    w = np.tile(recency_weights(games.date, as_of), 2)
    dev = np.linalg.solve(X.T @ (X * w[:, None]) + PRIOR_GAMES * np.eye(2 * n), X.T @ (w * y))
    est = prior + dev
    off, dfn = est[:n], est[n:]
    # keep the league average where the prior had it
    return off - off.mean() + prior[:n].mean(), dfn - dfn.mean() + prior[n:].mean()


def fit_pace(teams, games, lg_pace, as_of):
    """game possessions - league pace = home pace tendency + away pace tendency (ridge, preseason anchor)."""
    prior = teams.pace_prior.to_numpy()
    if games.empty:
        return prior.copy()
    n = len(teams)
    idx = {tid: i for i, tid in enumerate(teams.TEAM_ID)}
    h = games.home_id.map(idx).to_numpy()
    a = games.away_id.map(idx).to_numpy()
    X = np.zeros((len(games), n))
    X[np.arange(len(games)), h] = 1
    X[np.arange(len(games)), a] = 1
    y = games.poss.to_numpy() - lg_pace - X @ prior
    w = recency_weights(games.date, as_of)
    dev = np.linalg.solve(X.T @ (X * w[:, None]) + PACE_PRIOR_GAMES * np.eye(n), X.T @ (w * y))
    return prior + dev


def matchup(r_home, r_away, lg_ortg, lg_pace, rest_adj=0.0):
    """Projected possessions, scores, margin and total for one game from two rows of team ratings."""
    P = lg_pace + r_home.pace + r_away.pace
    home = (lg_ortg + r_home.off - r_away.dfn) * P / 100 + HOME_COURT / 2 + rest_adj / 2
    away = (lg_ortg + r_away.off - r_home.dfn) * P / 100 - HOME_COURT / 2 - rest_adj / 2
    return {"poss": P, "home_pts": home, "away_pts": away, "margin": home - away, "total": home + away}


def use_weighted_preseason(teams):
    """Swap in the weighted-statistics preseason power points when they exist."""
    if not os.path.exists(PRESEASON_PATH):
        print("Preseason ratings: combined win projections (run nba_preseason_power.py to use weighted statistics).")
        return teams
    pre_df = pd.read_csv(PRESEASON_PATH, encoding="utf-8-sig")
    pre = pre_df.set_index("TEAM_NAME").power_points
    labels = {"combined": "combined model (Version A + B)", "weighted": "weighted statistics (20 stats)",
              "compact": "compact statistics (7 stats)"}
    used = str(pre_df.model_used.iloc[0]) if "model_used" in pre_df else "weighted"
    if used.startswith("combined_"):
        stats = used.split("_")[1]
        label = f"blend of combined model and {stats} statistics"
    else:
        label = labels.get(used, used)
    missing = set(teams.TEAM_NAME) - set(pre.index)
    if missing:
        print(f"Preseason file is missing {missing}; using combined win projections instead.")
        return teams
    teams = teams.copy()
    teams["preseason_rating"] = teams.TEAM_NAME.map(pre)
    print(f"Preseason ratings: {label} ({PRESEASON_PATH}).")
    return teams


def write_team_ratings(wb, ratings, lg_o, lg_p, label):
    """Replace the Team Ratings sheet (script-owned) with this run's offense/defense/pace."""
    if "Team Ratings" in wb.sheetnames:
        ws = wb["Team Ratings"]
        ws.delete_rows(1, ws.max_row)
    else:
        ws = wb.create_sheet("Team Ratings")
    headers = ["Team", "Off rating", "Def rating", "Net rating", "Pace", "Off rank", "Def rank",
               "Pace rank", "Power points"]
    cols = ["TEAM_NAME", "off_rtg", "def_rtg", "net_rtg", "pace", "off_rank", "def_rank", "pace_rank", "rating"]
    ws.append(headers)
    for row in ratings[cols].itertuples(index=False):
        ws.append([v.item() if hasattr(v, "item") else v for v in row])
    notes = [("League offensive rating", round(lg_o, 2)), ("League pace", round(lg_p, 2)), ("As of", label),
             ("Off rating", "points scored per 100 possessions, opponent-adjusted"),
             ("Def rating", "points allowed per 100 possessions (lower is better)"),
             ("Net rating", "off minus def; per 100 possessions"),
             ("Pace", "expected possessions per game against a league-average-pace opponent"),
             ("Power points", "net rating at league pace = points better than average per game")]
    for i, (k, v) in enumerate(notes, 1):
        ws.cell(i, 11).value, ws.cell(i, 12).value = k, v
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["K"].width = 24
    ws.column_dimensions["L"].width = 16


def add_matchup_sheet(wb):
    """Create the Matchup Projection sheet once (formulas only), so later runs never touch it."""
    if "Matchup Projection" in wb.sheetnames:
        return
    from openpyxl.styles import Font
    from openpyxl.worksheet.datavalidation import DataValidation
    ws = wb.create_sheet("Matchup Projection")
    T = "'Team Ratings'!"

    def look(team_cell, col):
        return f"INDEX({T}${col}$2:${col}$31,MATCH({team_cell},{T}$A$2:$A$31,0))"

    lgO, lgP = f"{T}$L$1", f"{T}$L$2"
    rest = '(IF($B$24="Yes",$B$25,0)-IF($B$23="Yes",$B$25,0))'   # positive favors home
    ok = 'AND($B$3<>"",$B$4<>"",$B$3<>$B$4)'
    cells = {
        "A1": "Matchup Projection", "A2": "Team",
        "A3": "Home team", "B3": "Boston Celtics",
        "C3": f'=IFERROR("Off #"&{look("$B$3","F")}&"  Def #"&{look("$B$3","G")}&"  Pace #"&{look("$B$3","H")},"")',
        "A4": "Away team", "B4": "New York Knicks",
        "C4": f'=IFERROR("Off #"&{look("$B$4","F")}&"  Def #"&{look("$B$4","G")}&"  Pace #"&{look("$B$4","H")},"")',
        "A6": "Projected possessions",
        "B6": f'=IF({ok},{look("$B$3","E")}+{look("$B$4","E")}-{lgP},"")',
        "C6": "Both teams' pace combined",
        "A7": "Projected home score",
        "B7": f'=IF(ISNUMBER(B6),({look("$B$3","B")}+{look("$B$4","C")}-{lgO})*B6/100+$B$22/2+{rest}/2,"")',
        "A8": "Projected away score",
        "B8": f'=IF(ISNUMBER(B6),({look("$B$4","B")}+{look("$B$3","C")}-{lgO})*B6/100-$B$22/2-{rest}/2,"")',
        "A9": "Projected margin (home)", "B9": '=IF(ISNUMBER(B7),B7-B8,"")',
        "C9": "Positive = home team favored; adjusted for pace",
        "A10": "Projected total", "B10": '=IF(ISNUMBER(B7),B7+B8,"")',
        "A11": "Model line",
        "B11": '=IF(ISNUMBER(B9),IF(ROUND(B9*2,0)/2=0,"Pick\'em",IF(B9>0,$B$3&" -"&TEXT(ROUND(B9*2,0)/2,"0.0"),'
               '$B$4&" -"&TEXT(ROUND(-B9*2,0)/2,"0.0")))&"   O/U "&TEXT(ROUND(B10*2,0)/2,"0.0"),"")',
        "A12": "Win probability",
        "B12": '=IF(ISNUMBER(B9),$B$3&" "&TEXT(NORMSDIST(B9/$B$26),"0%")&"  /  "&$B$4&" "&TEXT(1-NORMSDIST(B9/$B$26),"0%"),"")',
        "A14": "Compare to a sportsbook (optional)",
        "A15": "Book spread for home team", "C15": "e.g. -3.5 if the home team is favored by 3.5",
        "A16": "Book total",
        "A17": "Spread difference", "B17": '=IF(AND(ISNUMBER(B9),ISNUMBER(B15)),B9+B15,"")',
        "C17": "Model margin minus book margin; positive = model rates the home team higher",
        "A18": "Total difference", "B18": '=IF(AND(ISNUMBER(B10),ISNUMBER(B16)),B10-B16,"")',
        "C18": "Model total minus book total; positive = model projects more points",
        "A21": "Settings",
        "A22": "Home court advantage (points)", "B22": HOME_COURT,
        "A23": "Home team on a back-to-back?", "B23": "No",
        "A24": "Away team on a back-to-back?", "B24": "No",
        "A25": "Back-to-back penalty (points)", "B25": B2B_PENALTY,
        "A26": "Game margin std. deviation", "B26": GAME_SD,
    }
    for k, v in cells.items():
        ws[k] = v
    for k in ["A1", "A2", "A14", "A21"]:
        ws[k].font = Font(bold=True, size=14 if k == "A1" else 11)
    for k in ["B6", "B7", "B8", "B9", "B10", "B17", "B18"]:
        ws[k].number_format = "0.0"
    teams_src = "Lists!$A$2:$A$31" if "Lists" in wb.sheetnames else f"{T}$A$2:$A$31"
    dv = DataValidation(type="list", formula1=teams_src, allow_blank=False)
    yn = DataValidation(type="list", formula1='"Yes,No"', allow_blank=False)
    ws.add_data_validation(dv)
    ws.add_data_validation(yn)
    dv.add("B3:B4")
    yn.add("B23:B24")
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 60


def update_calculator(table, label, ratings, lg_o, lg_p):
    """Write this run's numbers into the workbook. Only script-owned cells/sheets are changed."""
    if not os.path.exists(CALCULATOR_PATH):
        return
    from openpyxl import load_workbook
    try:
        wb = load_workbook(CALCULATOR_PATH)
    except PermissionError:
        print(f"Couldn't open {CALCULATOR_PATH} - close it in Excel and run again.")
        return
    ws = wb["Power Rankings"]
    headers = [ws.cell(1, c).value for c in range(1, 11)]
    if headers != list(table.columns):
        print(f"{CALCULATOR_PATH}: Power Rankings columns don't match, so it wasn't updated.")
        return
    for r in range(2, 32):
        for c in range(1, 11):
            ws.cell(r, c).value = None
    for r, row in enumerate(table.itertuples(index=False), 2):
        for c, v in enumerate(row, 1):
            ws.cell(r, c).value = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
    if "Spread Calculator" in wb.sheetnames:
        wb["Spread Calculator"]["C1"] = f"Ratings: {label}"
    write_team_ratings(wb, ratings, lg_o, lg_p, label)
    add_matchup_sheet(wb)
    wb.calculation.fullCalcOnLoad = True  # Excel/Sheets recompute the calculator when it opens
    try:
        wb.save(CALCULATOR_PATH)
        print(f"Updated {CALCULATOR_PATH}.")
    except PermissionError:
        print(f"Couldn't save {CALCULATOR_PATH} - close it in Excel and run again.")


def chart(rank, path, label):
    r = rank.sort_values("rating")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(r.TEAM_NAME, r.rating, color=np.where(r.rating >= 0, "#3b6ea5", "#c0504d"))
    ax.axvline(0, color="#888", lw=1)
    for i, (v, rk) in enumerate(zip(r.rating, r["rank"])):
        ax.text(v + (0.15 if v >= 0 else -0.15), i, f"#{rk}  {v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Points better than an average team (neutral court)")
    ax.set_title(f"NBA power rankings, {label}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def log_game_projections(sched, teams, lg_o, lg_p, today):
    """Save projections for the next two weeks of unplayed games, so nba_grade.py can grade them later."""
    if sched is None:
        return
    up = sched[sched.home_pts.isna()].copy()
    up["date"] = pd.to_datetime(up.date, errors="coerce")
    up = up[(up.date >= today) & (up.date < today + pd.Timedelta(days=14))]
    if up.empty:
        return
    r = teams.set_index("TEAM_ID")[["TEAM_NAME", "off", "dfn", "pace_adj"]].rename(columns={"pace_adj": "pace"})
    rows = []
    for g in up.itertuples():
        m = matchup(r.loc[g.home_id], r.loc[g.away_id], lg_o, lg_p, g.rest_adj)
        rows.append({"as_of": today.date().isoformat(), "game_id": g.game_id, "date": g.date.date().isoformat(),
                     "home": r.TEAM_NAME[g.home_id], "away": r.TEAM_NAME[g.away_id],
                     "proj_margin": round(m["margin"], 2), "proj_total": round(m["total"], 2),
                     "home_win_prob": round(norm.cdf(m["margin"] / GAME_SD), 3)})
    new = pd.DataFrame(rows)
    new.to_csv(os.path.join(OUT_DIR, "game_projections.csv"), index=False, encoding="utf-8-sig")
    path = os.path.join(OUT_DIR, "game_projection_history.csv")
    if os.path.exists(path):
        old = pd.read_csv(path, encoding="utf-8-sig", dtype={"game_id": str})
        new = pd.concat([old[old.as_of != new.as_of.iloc[0]], new], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    today = pd.Timestamp(dt.date.today())

    teams = use_weighted_preseason(S.load_teams())
    sched = S.fetch_schedule(set(teams.TEAM_ID))
    if sched is None:
        played = pd.DataFrame(columns=["home_id", "away_id", "home_pts", "away_pts", "date", "rest_adj"])
    else:
        sched["game_id"] = sched.game_id.astype(str)
        sched = back_to_back_adjustment(sched)
        played = sched[sched.home_pts.notna()]

    # offense, defense and pace
    teams, lg_o, lg_p = add_priors(teams, fetch_last_season_advanced())
    games = played.copy()
    if len(games):
        games["game_id"] = games.game_id.astype(str).str.zfill(10)
        games = games.merge(fetch_box_log(), on="game_id", how="left")
        has_poss = games.poss.notna()
        if has_poss.any():
            # this season's league levels, blended with last season's until enough games are in
            g = games[has_poss]
            w = len(g) / (len(g) + 60)
            season_o = 100 * (g.home_pts.sum() + g.away_pts.sum()) / (2 * g.poss.sum())
            lg_o, lg_p = w * season_o + (1 - w) * lg_o, w * g.poss.mean() + (1 - w) * lg_p
        games["poss"] = games.poss.fillna(lg_p)
        for c in ["home_pts", "away_pts", "rest_adj"]:
            games[c] = games[c].astype(float)
    else:
        has_poss = pd.Series(dtype=bool)
    teams["off"], teams["dfn"] = fit_off_def(teams, games, lg_o, today)
    teams["pace_adj"] = fit_pace(teams, games[has_poss] if len(games) else games, lg_p, today)
    teams["rating"] = (teams.off + teams.dfn) * lg_p / 100

    ratings = teams.assign(off_rtg=lg_o + teams.off, def_rtg=lg_o - teams.dfn,
                           net_rtg=teams.off + teams.dfn, pace=lg_p + teams.pace_adj)
    ratings["off_rank"] = ratings.off_rtg.rank(ascending=False, method="min").astype(int)
    ratings["def_rank"] = ratings.def_rtg.rank(ascending=True, method="min").astype(int)
    ratings["pace_rank"] = ratings.pace.rank(ascending=False, method="min").astype(int)
    ratings = ratings.sort_values("net_rtg", ascending=False).reset_index(drop=True)
    ratings.insert(0, "rank", ratings.index + 1)
    rcols = ["rank", "TEAM_NAME", "conf", "off_rtg", "def_rtg", "net_rtg", "pace",
             "off_rank", "def_rank", "pace_rank", "rating"]
    ratings = ratings[rcols].round(1)
    ratings.to_csv(os.path.join(OUT_DIR, "team_ratings.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"league_off_rtg": [round(lg_o, 2)], "league_pace": [round(lg_p, 2)],
                  "as_of": [today.date().isoformat()], "games": [len(played)]}).to_csv(
        os.path.join(OUT_DIR, "league_levels.csv"), index=False)

    # records
    if len(played):
        rec = pd.concat([
            pd.DataFrame({"TEAM_ID": played.home_id, "won": played.home_pts > played.away_pts}),
            pd.DataFrame({"TEAM_ID": played.away_id, "won": played.away_pts > played.home_pts}),
        ]).groupby("TEAM_ID").won.agg(["sum", "size"])
        teams["W"] = teams.TEAM_ID.map(rec["sum"]).fillna(0).astype(int)
        teams["L"] = teams.TEAM_ID.map(rec["size"]).fillna(0).astype(int) - teams.W
    else:
        teams["W"], teams["L"] = 0, 0

    rank = teams.sort_values("rating", ascending=False).reset_index(drop=True)
    rank.insert(0, "rank", rank.index + 1)
    rank["preseason_rank"] = rank.preseason_rating.rank(ascending=False, method="first").astype(int)
    rank["change_since_preseason"] = rank.rating - rank.preseason_rating
    # full-season win pace this rating implies against an average schedule
    rank["pace_82"] = norm.cdf(rank.rating / GAME_SD) * 82

    # movement since the last run on an earlier day
    hist_path = os.path.join(OUT_DIR, "power_history.csv")
    hist = pd.read_csv(hist_path, encoding="utf-8-sig") if os.path.exists(hist_path) else pd.DataFrame()
    today_str = today.date().isoformat()
    rank["rank_change"] = pd.array([pd.NA] * len(rank), dtype="Int64")
    if not hist.empty:
        earlier = hist[hist.as_of < today_str]
        if not earlier.empty:
            prev = earlier[earlier.as_of == earlier.as_of.max()].set_index("TEAM_NAME")["rank"]
            rank["rank_change"] = (rank.TEAM_NAME.map(prev) - rank["rank"]).astype("Int64")
        hist = hist[hist.as_of != today_str]

    snap = rank[["rank", "TEAM_NAME", "conf", "W", "L", "rating"]].merge(
        ratings[["TEAM_NAME", "off_rtg", "def_rtg", "net_rtg", "pace"]], on="TEAM_NAME").assign(as_of=today_str)
    pd.concat([hist, snap], ignore_index=True).to_csv(hist_path, index=False, encoding="utf-8-sig")

    cols = ["rank", "rank_change", "TEAM_NAME", "conf", "W", "L", "rating", "pace_82",
            "preseason_rank", "change_since_preseason"]
    rank[cols].round(1).to_csv(os.path.join(OUT_DIR, "power_rankings.csv"), index=False, encoding="utf-8-sig")

    label = "preseason" if played.empty else f"{today.date()} ({len(played)} games played)"
    print(f"\nNBA power rankings, {label}")
    print("rating = points better than an average team on a neutral court; "
          "rank_change = spots moved since the last run\n")
    print(rank[cols].round(1).to_string(index=False))
    print(f"\nOffense, defense and pace (league: {lg_o:.1f} points per 100, {lg_p:.1f} possessions)\n")
    print(ratings.drop(columns="conf").to_string(index=False))
    chart(rank, os.path.join(OUT_DIR, "power_rankings.png"), label)
    update_calculator(rank[cols].round(1), label, ratings, lg_o, lg_p)
    log_game_projections(sched, teams, lg_o, lg_p, today)
    print(f"\nSaved results to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
