"""
NBA model grading

Checks the projections against what actually happened, so you can see whether the model is any
good and which settings need adjusting.

Player props (from nba_player_props.py's history files vs. box scores):
  - MAE: average miss per stat, next to a simple baseline (the player's season average so far).
    The model should beat the baseline; if it doesn't for a stat, that stat needs work.
  - Bias: average projection minus actual. Positive = projecting too high.
  - Range check: how often the result fell under the median (target ~50%), below the 10th
    percentile and above the 90th (each target ~10%; lower for low-count stats like steals and
    blocks, where many results land exactly on the range edges). If results fall outside the
    range much more often than that, the ranges are too narrow; much less often, too wide.
  - Players projected who didn't play are counted separately (props are voided for them).

Games (from nba_power_ratings.py's game_projection_history.csv vs. final scores), using the latest
projection made on or before game day:
  - Margin MAE vs. a baseline of "home team by HOME_COURT", straight-up winner accuracy, Brier score for
    win probabilities (lower is better; 0.25 is a coin flip), total MAE and bias, and the share of
    games that went over the model's projected total (about 50% if the totals don't lean either way).

Tuning test (--tune): replays this season day by day. Each day's games are predicted using only
earlier games, once for each value of a setting: the preseason anchors, home court, the back-to-back
penalty and the recency half-life. Results are reported, not applied - change PRIOR_GAMES,
PACE_PRIOR_GAMES, B2B_PENALTY or HALF_LIFE_DAYS in nba_power_ratings.py, or HOME_COURT in
nba_season_sim.py, yourself if a different value keeps winning over several weeks.

Outputs (output_grades/): player_grades.csv, player_grades_daily.csv, game_grades.csv,
game_results.csv, anchor_tuning.csv (with --tune). If NBA_Spread_Calculator.xlsx is in this folder,
the Model Grades sheet is replaced with the summary. Close the workbook first.

Usage:
  python3 nba_grade.py          # grade everything so far
  python3 nba_grade.py --tune   # also run the tuning test (best after ~4 weeks of games)
"""

import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_player_props as P
import nba_power_ratings as PR
import nba_season_sim as S

OUT_DIR = "output_grades"
STAT_ORDER = ["MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV", "PRA", "P+R", "P+A", "R+A"]
LOG_COLS = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PM": "FG3M", "STL": "STL", "BLK": "BLK", "TOV": "TOV"}
LOW_COUNT = {"3PM", "STL", "BLK", "TOV"}
BASELINE_MIN_GAMES = 3
ANCHORS = [1, 5, 10, 20, 30, 40]   # 1 = almost no preseason anchor
PACE_ANCHORS = [1, 5, 10, 20, 30]
HOME_COURTS = [1.0, 1.5, 1.8, 2.3, 3.0]   # points
B2B_PENALTIES = [0.0, 1.0, 1.8, 2.5, 3.5]   # points
HALF_LIVES = [20, 30, 60, 120]             # days
TUNE_SKIP_DAYS = 7        # the first week has too few earlier games to predict from


# ------------------------------------------------------------ player props
def actual_player_stats(log):
    a = log[["PLAYER_ID", "GAME_DATE", "MIN"] + P.STATS].copy()
    a = a.rename(columns={v: k for k, v in LOG_COLS.items()})
    a["PRA"], a["P+R"] = a.PTS + a.REB + a.AST, a.PTS + a.REB
    a["P+A"], a["R+A"] = a.PTS + a.AST, a.REB + a.AST
    a = a.sort_values(["PLAYER_ID", "GAME_DATE"])
    # baseline: his average in this season's earlier games
    for s in STAT_ORDER:
        prior = a.groupby("PLAYER_ID")[s].transform(lambda x: x.shift().expanding().mean())
        n = a.groupby("PLAYER_ID").cumcount()
        a[f"base_{s}"] = prior.where(n >= BASELINE_MIN_GAMES)
    a["date"] = a.GAME_DATE.dt.date.astype(str)
    return a


def grade_players(log):
    hist_path = os.path.join(P.OUT_DIR, "projection_history.csv")
    if not os.path.exists(hist_path):
        print("No player projections saved yet (run nba_player_props.py on game days).")
        return None, None
    proj = pd.read_csv(hist_path, encoding="utf-8-sig")
    actual = actual_player_stats(log)
    played_dates = set(actual.date)
    proj = proj[proj.date.isin(played_dates)]
    if proj.empty:
        print("No completed games yet for the saved player projections.")
        return None, None
    m = proj.merge(actual, on=["date", "PLAYER_ID"], how="left", suffixes=("_proj", ""))
    dnp = m.MIN.isna()
    print(f"Player projections graded: {(~dnp).sum()} player-games ({dnp.sum()} projected players didn't play).")
    m = m[~dnp]

    rh_path = os.path.join(P.OUT_DIR, "range_history.csv")
    ranges = pd.read_csv(rh_path, encoding="utf-8-sig") if os.path.exists(rh_path) else None

    rows, daily = [], []
    for s in STAT_ORDER:
        err = m[f"{s}_proj"] - m[s]
        has_base = m[f"base_{s}"].notna()
        row = {"stat": s, "n": len(m), "MAE": err.abs().mean(), "bias": err.mean(),
               "baseline_n": int(has_base.sum()),
               "MAE_same_games": err[has_base].abs().mean(),
               "baseline_MAE": (m[f"base_{s}"] - m[s])[has_base].abs().mean()}
        if ranges is not None and s != "MIN":
            r = ranges[ranges.Stat == s].merge(m[["date", "PLAYER_ID", s]], on=["date", "PLAYER_ID"])
            if len(r):
                row["pct_under_median"] = 100 * ((r[s] < r.Median).mean() + 0.5 * (r[s] == r.Median).mean())
                row["pct_below_P10"] = 100 * (r[s] < r.P10).mean()
                row["pct_above_P90"] = 100 * (r[s] > r.P90).mean()
        rows.append(row)
        d = m.assign(abs_err=err.abs()).groupby("date").agg(n=("abs_err", "size"), MAE=("abs_err", "mean"))
        daily.append(d.reset_index().assign(stat=s))
    g = pd.DataFrame(rows)
    g["model_beats_baseline"] = np.where(g.baseline_n == 0, "",
                                         np.where(g.MAE_same_games < g.baseline_MAE, "yes", "no"))
    g["range_verdict"] = g.apply(range_verdict, axis=1)
    return g.round(2), pd.concat(daily, ignore_index=True).round(2)


def range_verdict(r):
    if pd.isna(r.get("pct_below_P10", np.nan)) or r.n < 100:
        return "not enough games yet" if r.stat != "MIN" else ""
    outside = r.pct_below_P10 + r.pct_above_P90
    if outside > 28:
        return "ranges too narrow"
    if outside < (6 if r.stat in LOW_COUNT else 12):
        return "ranges too wide"
    if abs(r.pct_under_median - 50) > 7:
        return "median too " + ("high" if r.pct_under_median > 50 else "low")
    return "ok"


# ------------------------------------------------------------------ games
def grade_games(sched):
    path = os.path.join(PR.OUT_DIR, "game_projection_history.csv")
    if not os.path.exists(path):
        print("No game projections saved yet (nba_power_ratings.py saves them on each run).")
        return None, None
    h = pd.read_csv(path, encoding="utf-8-sig", dtype={"game_id": str})
    h = h[h.as_of <= h.date].sort_values("as_of").groupby("game_id").tail(1)
    res = sched[sched.home_pts.notna()][["game_id", "home_pts", "away_pts"]]
    m = h.merge(res, on="game_id")
    if m.empty:
        print("No completed games yet for the saved game projections.")
        return None, None
    m["margin"] = m.home_pts - m.away_pts
    m["total"] = m.home_pts + m.away_pts
    m["home_won"] = (m.margin > 0).astype(int)
    m["margin_err"] = m.proj_margin - m.margin
    m["total_err"] = m.proj_total - m.total
    g = pd.DataFrame([{
        "games": len(m),
        "margin_MAE": m.margin_err.abs().mean(),
        "baseline_margin_MAE": (PR.HOME_COURT - m.margin).abs().mean(),
        "winner_pct": 100 * ((m.proj_margin > 0) == (m.margin > 0)).mean(),
        "brier": ((m.home_win_prob - m.home_won) ** 2).mean(),
        "total_MAE": m.total_err.abs().mean(),
        "total_bias": m.total_err.mean(),
        # share of games whose actual total went over the model's projected total (about 50% if unbiased)
        "pct_over_total": 100 * (m.total > m.proj_total).mean(),
    }]).round(3)
    print(f"Games graded: {len(m)}.")
    return g, m


# ------------------------------------------------------------ anchor test
def anchor_test(teams, sched):
    played = sched[sched.home_pts.notna()].copy()
    played["date"] = pd.to_datetime(played.date)
    if len(played) < 150:
        print(f"\nTuning test: only {len(played)} games played; results will be noisy until ~200+.")
    teams, lg_o0, lg_p0 = PR.add_priors(teams, PR.fetch_last_season_advanced())
    played["game_id"] = played.game_id.astype(str).str.zfill(10)
    played = played.merge(PR.fetch_box_log(), on="game_id", how="left")
    has_poss = played.poss.notna()
    for c in ["home_pts", "away_pts", "rest_adj"]:
        played[c] = played[c].astype(float)
    dates = sorted(played.date.unique())
    dates = [d for d in dates if d >= dates[0] + pd.Timedelta(days=TUNE_SKIP_DAYS)]

    orig = PR.PRIOR_GAMES, PR.PACE_PRIOR_GAMES, PR.HOME_COURT, PR.B2B_PENALTY, PR.HALF_LIFE_DAYS
    tests = [("efficiency", ANCHORS), ("pace", PACE_ANCHORS), ("home court", HOME_COURTS),
             ("back-to-back", B2B_PENALTIES), ("half-life", HALF_LIVES)]
    rows = []
    try:
        for kind, grid in tests:
            for v in grid:
                PR.PRIOR_GAMES = v if kind == "efficiency" else orig[0]
                PR.PACE_PRIOR_GAMES = v if kind == "pace" else orig[1]
                PR.HOME_COURT = v if kind == "home court" else orig[2]
                PR.B2B_PENALTY = v if kind == "back-to-back" else orig[3]
                PR.HALF_LIFE_DAYS = v if kind == "half-life" else orig[4]
                pl = played.assign(rest_adj=played.rest_adj * PR.B2B_PENALTY / orig[3])
                errs = []
                for d in dates:
                    before = pl[pl.date < d]
                    bp = before[has_poss[before.index]]
                    lg_o, lg_p = lg_o0, lg_p0
                    if len(bp):
                        w = len(bp) / (len(bp) + PR.LEAGUE_PRIOR_GAMES)
                        lg_o = w * 100 * (bp.home_pts.sum() + bp.away_pts.sum()) / (2 * bp.poss.sum()) + (1 - w) * lg_o0
                        lg_p = w * bp.poss.mean() + (1 - w) * lg_p0
                    g = before.assign(poss=before.poss.fillna(lg_p))
                    t = teams.copy()
                    t["off"], t["dfn"] = PR.fit_off_def(t, g, lg_o, d)
                    t["pace"] = PR.fit_pace(t, g[has_poss[g.index]], lg_p, d)
                    r = t.set_index("TEAM_ID")
                    for x in pl[pl.date == d].itertuples():
                        mm = PR.matchup(r.loc[x.home_id], r.loc[x.away_id], lg_o, lg_p, x.rest_adj)
                        errs.append((mm["margin"] - (x.home_pts - x.away_pts), mm["total"] - (x.home_pts + x.away_pts)))
                e = np.array(errs)
                current = v == orig[[t[0] for t in tests].index(kind)]
                rows.append({"test": kind, "value": v, "games_predicted": len(e),
                             "margin_MAE": np.abs(e[:, 0]).mean(), "total_MAE": np.abs(e[:, 1]).mean(),
                             "margin_bias": e[:, 0].mean(), "current_setting": "yes" if current else ""})
    finally:
        PR.PRIOR_GAMES, PR.PACE_PRIOR_GAMES, PR.HOME_COURT, PR.B2B_PENALTY, PR.HALF_LIFE_DAYS = orig
    out = pd.DataFrame(rows).round(3)
    for kind, col, name, unit in [("efficiency", "margin_MAE", "PRIOR_GAMES", "games"),
                                  ("pace", "total_MAE", "PACE_PRIOR_GAMES", "games"),
                                  ("home court", "margin_MAE", "HOME_COURT", "points"),
                                  ("back-to-back", "margin_MAE", "B2B_PENALTY", "points"),
                                  ("half-life", "margin_MAE", "HALF_LIFE_DAYS", "days")]:
        k = out[out.test == kind]
        best, cur = k.loc[k[col].idxmin()], k[k.current_setting == "yes"].iloc[0]
        gain = cur[col] - best[col]
        msg = "keep the current setting" if gain < 0.05 else f"consider {name} = {best.value:g} if it holds up"
        print(f"  {kind}: best {best.value:g} {unit} ({col} {best[col]:.2f}) vs current "
              f"{cur.value:g} ({cur[col]:.2f}) -> {msg}")
    return out


# ------------------------------------------------------------ pick history
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
LINES_PATH = os.path.join(S.DATA_DIR, "closing_lines.csv")
SEASON_START = pd.Timestamp("2026-10-20")   # a Tuesday; weeks run Tuesday to Monday like the Bet Summary
LINE_COLS = ["date", "home", "away", "provider", "home_close", "home_open", "home_odds", "away_odds",
             "total_close", "total_open", "over_odds", "under_odds"]


def _num(text):
    """'+1.5' -> 1.5, 'o220.5' -> 220.5, 'PK'/'EVEN' -> 0; None if missing."""
    if text is None:
        return None
    t = str(text).strip().lower().lstrip("ou")
    if t in ("pk", "even", "ev"):
        return 0.0
    try:
        return float(t)
    except ValueError:
        return None


def _get(d, *keys):
    """d[k1][k2]... with None for anything missing."""
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def fetch_lines(games, team_names):
    """
    Opening and closing spreads and totals (DraftKings, from ESPN's public game summaries) for every
    game on these dates. Saved in data/closing_lines.csv so each game is only looked up once.
    """
    import requests
    cache = pd.read_csv(LINES_PATH) if os.path.exists(LINES_PATH) else pd.DataFrame(columns=LINE_COLS)
    have = set(zip(cache.date.astype(str), cache.home))
    need = {d for d, h in zip(games.date.astype(str), games.home) if (d, h) not in have}
    by_nick = {n.split()[-1]: n for n in team_names}
    rows = []
    for d in sorted(need):
        try:
            sb = requests.get(f"{ESPN}/scoreboard", params={"dates": d.replace("-", "")},
                              headers=S.BROWSER_HEADERS, timeout=30).json()
            for ev in sb.get("events", []):
                comp = ev["competitions"][0]
                if not comp.get("status", {}).get("type", {}).get("completed"):
                    continue
                if (d, by_nick.get(next(c for c in comp["competitors"] if c["homeAway"] == "home")["team"]["displayName"].split()[-1])) in have:
                    continue
                side = {c["homeAway"]: by_nick.get(c["team"]["displayName"].split()[-1]) for c in comp["competitors"]}
                pc = requests.get(f"{ESPN}/summary", params={"event": ev["id"]},
                                  headers=S.BROWSER_HEADERS, timeout=30).json().get("pickcenter", [])
                if not pc:
                    continue
                ps, tot = pc[0].get("pointSpread", {}), pc[0].get("total", {})
                rows.append({"date": d, "home": side.get("home"), "away": side.get("away"),
                             "provider": pc[0].get("provider", {}).get("name", ""),
                             "home_close": _num(_get(ps, "home", "close", "line")),
                             "home_open": _num(_get(ps, "home", "open", "line")),
                             "home_odds": _num(_get(ps, "home", "close", "odds")),
                             "away_odds": _num(_get(ps, "away", "close", "odds")),
                             "total_close": _num(_get(tot, "over", "close", "line")),
                             "total_open": _num(_get(tot, "over", "open", "line")),
                             "over_odds": _num(_get(tot, "over", "close", "odds")),
                             "under_odds": _num(_get(tot, "under", "close", "odds"))})
        except Exception as e:
            print(f"Couldn't get lines for {d} ({e.__class__.__name__}); will try again next run.")
            continue
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(LINES_PATH, index=False)
        print(f"Lines: added {len(rows)} games from ESPN ({len(cache)} saved).")
    return cache


def _risk_units(odds, result):
    odds = -110.0 if pd.isna(odds) else odds
    risk = -odds / 100 if odds < 0 else 1.0
    win = 1.0 if odds < 0 else odds / 100
    return risk, {"W": win, "L": -risk, "P": 0.0}[result]


def pick_history(games):
    """
    The model's pick for every graded game against the closing line, whether or not it was bet.
    Spread: the side the model likes better than the line (underdog if it projects the underdog to
    do better than the line says, favorite if it projects the favorite to win by more). Total: Over if
    the projected total is above the closing total, Under if below. Each row is written from the pick's
    side, like the Bet Log: Line, Model and Open are the pick's spread (or the total), Final is the
    pick's margin (or the total). Line move = points the line moved toward the pick from open to close.
    """
    lines = fetch_lines(games, set(games.home) | set(games.away))
    m = games.assign(date=games.date.astype(str)).merge(lines, on=["date", "home", "away"], how="inner")
    rows = []
    for g in m.itertuples():
        week = (pd.Timestamp(g.date) - SEASON_START).days // 7 + 1
        base = {"Week": week, "Date": g.date, "Away": g.away, "Home": g.home}
        if pd.notna(g.home_close) and g.proj_margin + g.home_close != 0:
            home = g.proj_margin + g.home_close > 0          # model margin beats the market's margin
            s = 1 if home else -1
            line, opn = s * g.home_close, s * (g.home_open if pd.notna(g.home_open) else g.home_close)
            final = s * g.margin
            res = "P" if final + line == 0 else ("W" if final + line > 0 else "L")
            risk, units = _risk_units(g.home_odds if home else g.away_odds, res)
            rows.append({**base, "Type": "Spread", "Pick": g.home if home else g.away,
                         "Side": "Favorite" if line < 0 else ("Underdog" if line > 0 else "Pick'em"),
                         "Venue": "Home" if home else "Away", "Line": line,
                         "Odds": g.home_odds if home else g.away_odds, "Model": -s * g.proj_margin,
                         "Open": opn, "Final": final, "Edge": abs(g.proj_margin + g.home_close),
                         "Risk": risk, "Result": res, "Units": units, "Line move": opn - line})
        if pd.notna(g.total_close) and g.proj_total != g.total_close:
            over = g.proj_total > g.total_close
            opn = g.total_open if pd.notna(g.total_open) else g.total_close
            res = "P" if g.total == g.total_close else ("W" if (g.total > g.total_close) == over else "L")
            risk, units = _risk_units(g.over_odds if over else g.under_odds, res)
            rows.append({**base, "Type": "Total", "Pick": "Over" if over else "Under",
                         "Side": "Over" if over else "Under", "Venue": "", "Line": g.total_close,
                         "Odds": g.over_odds if over else g.under_odds, "Model": g.proj_total,
                         "Open": opn, "Final": g.total, "Edge": abs(g.proj_total - g.total_close),
                         "Risk": risk, "Result": res, "Units": units,
                         "Line move": (g.total_close - opn) if over else (opn - g.total_close)})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Date", "Home", "Type"]).round(2)
        for t in ["Spread", "Total"]:
            k = out[out.Type == t]
            w, l, p = (k.Result == "W").sum(), (k.Result == "L").sum(), (k.Result == "P").sum()
            print(f"Model picks vs closing lines, {t.lower()}s: {w}-{l}-{p}, {k.Units.sum():+.2f} units")
    missing = len(games) - m.shape[0]
    if missing:
        print(f"  ({missing} graded games have no lines yet.)")
    return out


EDGE_BUCKETS = [("Under 3", 0, 3), ("3 to 4.5", 3, 5), ("5 to 6.5", 5, 7), ("7 or more", 7, 1000)]
SUMMARY_COLS = ["Picks", "W", "L", "P", "Win %", "Units", "Return on risk", "Line moved toward pick", "Avg line move"]


def _summary_row(label, k):
    """One table row: record, units, return on risk, and how often the line moved toward the pick."""
    w, l, p = (k.Result == "W").sum(), (k.Result == "L").sum(), (k.Result == "P").sum()
    if len(k) == 0:
        return [label, 0, 0, 0, 0, "", "", "", "", ""]
    return [label, len(k), w, l, p, f"{100 * w / max(w + l, 1):.1f}%", f"{k.Units.sum():+.2f}",
            f"{100 * k.Units.sum() / k.Risk.sum():+.1f}%", f"{100 * (k['Line move'] > 0).mean():.0f}%",
            f"{k['Line move'].mean():+.2f}"]


def pick_summary(picks):
    """The Pick Summary tab: a list of rows (title, header and table rows, with blank rows between)."""
    sp, to = picks[picks.Type == "Spread"], picks[picks.Type == "Total"]
    groups = [("Spreads", sp), ("  Favorites", sp[sp.Side == "Favorite"]), ("  Underdogs", sp[sp.Side == "Underdog"]),
              ("  Home picks", sp[sp.Venue == "Home"]), ("  Away picks", sp[sp.Venue == "Away"]),
              ("Totals", to), ("  Overs", to[to.Side == "Over"]), ("  Unders", to[to.Side == "Under"])]
    rows = [["Model pick history against closing lines (every game, bet or not)"],
            [f"Picks: spread = the side the model likes better than the closing line; total = Over if the "
             f"projected total is above the closing total, Under if below. Lines: DraftKings close via ESPN. "
             f"Graded at closing odds. Line move = points the line moved toward the pick from open to close. "
             f"Through {picks.Date.max()}."], []]

    def table(title, first, items):
        rows.extend([[title], [first] + SUMMARY_COLS] + [_summary_row(lab, k) for lab, k in items] + [[]])

    table("Overall", "Type", [("All picks", picks), ("Spreads", sp), ("Totals", to)])
    table("By side", "Side", groups[1:5] + groups[6:])
    for name, k in groups:
        if name.strip() in ("Home picks", "Away picks"):
            continue
        table(f"{name.strip()} by model edge (points between the line and the model)", "Edge",
              [(lab, k[(k.Edge >= lo) & (k.Edge < hi)]) for lab, lo, hi in EDGE_BUCKETS])
    moved = lambda k: [("Moved toward pick", k[k["Line move"] > 0]), ("No move", k[k["Line move"] == 0]),
                       ("Moved away from pick", k[k["Line move"] < 0])]
    table("Spreads by line movement (open to close)", "Line", moved(sp))
    table("Totals by line movement (open to close)", "Line", moved(to))
    weeks = []
    for wk, k in picks.groupby("Week"):
        start = SEASON_START + pd.Timedelta(days=7 * (wk - 1))
        weeks.append((f"{start:%m/%d} - {start + pd.Timedelta(days=6):%m/%d}", k))
    table("Weekly, spreads (Tuesday to Monday)", "Week", [(lab, k[k.Type == "Spread"]) for lab, k in weeks])
    table("Weekly, totals (Tuesday to Monday)", "Week", [(lab, k[k.Type == "Total"]) for lab, k in weeks])
    return rows


# ------------------------------------------------------------------ main
def write_workbook(pg, gg):
    if not os.path.exists(PR.CALCULATOR_PATH) or (pg is None and gg is None):
        return
    from openpyxl import load_workbook
    try:
        wb = load_workbook(PR.CALCULATOR_PATH)
    except PermissionError:
        print(f"Couldn't open {PR.CALCULATOR_PATH} - close it in Excel and run again.")
        return
    if "Model Grades" in wb.sheetnames:
        ws = wb["Model Grades"]
        ws.delete_rows(1, ws.max_row)
    else:
        ws = wb.create_sheet("Model Grades")
    for title, df in [("Games", gg), ("Player projections", pg)]:
        if df is None:
            continue
        ws.append([title])
        ws.append(list(df.columns))
        for row in df.itertuples(index=False):
            ws.append([None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row])
        ws.append([])
    ws.column_dimensions["A"].width = 16
    try:
        wb.save(PR.CALCULATOR_PATH)
        print(f"Updated {PR.CALCULATOR_PATH} (Model Grades).")
    except PermissionError:
        print(f"Couldn't save {PR.CALCULATOR_PATH} - close it in Excel and run again.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    teams = PR.use_weighted_preseason(S.load_teams())
    sched = S.fetch_schedule(set(teams.TEAM_ID))
    if sched is None:
        raise SystemExit("Couldn't get the schedule.")
    sched["game_id"] = sched.game_id.astype(str).str.zfill(10)
    sched = PR.back_to_back_adjustment(sched)

    gg, games = grade_games(sched)
    if gg is not None:
        gg.to_csv(os.path.join(OUT_DIR, "game_grades.csv"), index=False)
        games.to_csv(os.path.join(OUT_DIR, "game_results.csv"), index=False, encoding="utf-8-sig")
        print("\nGames\n" + gg.to_string(index=False))
        picks = pick_history(games)
        if len(picks):
            picks.to_csv(os.path.join(OUT_DIR, "pick_history.csv"), index=False, encoding="utf-8-sig")
            with open(os.path.join(OUT_DIR, "pick_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerows(pick_summary(picks))

    pg, daily = grade_players(P.season_game_log())
    if pg is not None:
        pg.to_csv(os.path.join(OUT_DIR, "player_grades.csv"), index=False)
        daily.to_csv(os.path.join(OUT_DIR, "player_grades_daily.csv"), index=False)
        print("\nPlayer projections (MAE_same_games and baseline_MAE use only games where the player "
              f"had {BASELINE_MIN_GAMES}+ earlier games)\n" + pg.to_string(index=False))

    if "--tune" in sys.argv:
        print("\nTuning test: anchors, home court, back-to-backs and recency (each day predicted from earlier games only):")
        at = anchor_test(teams, sched)
        at.to_csv(os.path.join(OUT_DIR, "anchor_tuning.csv"), index=False)
        print(at.to_string(index=False))

    write_workbook(pg, gg)
    print(f"\nSaved results to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
