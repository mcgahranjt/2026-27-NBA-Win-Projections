"""
NBA preseason power ratings from weighted statistics

Predicts each team's power points for the coming season (points per game better than an average
team on a neutral court) from a broad set of statistics, with the weights learned from history.

Features, all known before opening night:

  This season's roster (each player's last three seasons, weighted 60/30/10, projected forward with
  aging curves, then averaged using projected minutes from a depth chart):
    roster_obpm      offensive Box Plus/Minus
    roster_dbpm      defensive Box Plus/Minus
    roster_ws48      win shares per 48 minutes
    roster_ts        true shooting % (vs. league average)
    roster_tov       turnover %          roster_ast   assist %
    roster_orb       offensive rebound % roster_drb   defensive rebound %
    roster_stl       steal %             roster_blk   block %
    star_power       average BPM of the two best players getting real minutes
    bench_depth      average BPM of the 6th-10th men
    roster_age       minutes-weighted age

  Last season's team (multiplied by returning minutes share, so a rebuilt roster counts for less):
    prev_net_x_cont       net rating
    efg_diff_x_cont       effective FG% minus opponents'
    tov_diff_x_cont       opponents' turnover % minus own
    oreb_diff_x_cont      offensive rebound % minus opponents'
    ftr_diff_x_cont       free-throw rate minus opponents'
    continuity            share of last season's minutes still on the roster
    opp3_luck             league-average opponent 3P% minus the team's (opponent 3P% is mostly luck,
                          so teams that "benefited" tend to fall back)

Target: next season's net rating. The weights are fit with ridge regression (which keeps any one
statistic from dominating), with the penalty chosen by leave-one-season-out cross-validation.

Backtest: each of 2021-22 to 2025-26 is predicted using only earlier seasons and compared with a
BPM-only version, to show whether the extra statistics help.

Needs the data folder from Versions A and B. Downloads (once, then cached):
  - Basketball-Reference advanced stats with all columns (19 pages, ~2 minutes)
  - NBA.com team Four Factors and opponent shooting by season

Usage:
  python3 nba_preseason_power.py
"""

import os
import time
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import nba_wins_version_b as B
from nba_season_sim import EAST

DATA_DIR = B.DATA_DIR
OUT_DIR = "output_preseason_power"
TARGET = B.TARGET                 # 2026 = 2026-27
LAST_COMPLETED = B.LAST_COMPLETED
FIRST_TARGET = 2011               # first season with prior-season team stats from NBA.com
TEST_SEASONS = range(2021, 2026)
PRIOR_MINUTES = 500               # shrinkage, as tuned in Version B
ALPHAS = np.logspace(-1, 3, 25)

RATE_STATS = ["TS%", "TOV%", "AST%", "ORB%", "DRB%", "STL%", "BLK%"]
IMPACT_PRIORS = {"OBPM": -1.5, "DBPM": -0.5, "BPM": -2.0, "WS/48": 0.05}   # where small samples get pulled
ROOKIE_IMPACT = {"OBPM": -1.8, "DBPM": -0.7, "BPM": -2.5, "WS/48": 0.04}
BR_KEEP = ["Player", "Age", "Team", "G", "MP"] + RATE_STATS + list(IMPACT_PRIORS)

FEATURES = ["roster_obpm", "roster_dbpm", "roster_ws48", "roster_ts", "roster_tov", "roster_ast",
            "roster_orb", "roster_drb", "roster_stl", "roster_blk", "star_power", "bench_depth",
            "roster_age", "prev_net_x_cont", "efg_diff_x_cont", "tov_diff_x_cont", "oreb_diff_x_cont",
            "ftr_diff_x_cont", "continuity", "opp3_luck"]
BPM_ONLY = ["roster_bpm"]
# De-duplicated set: BPM already contains shooting, rebounding, assists, steals, blocks and turnovers,
# win shares overlap BPM, and the Four Factors overlap last season's net rating. Keep one measure of
# each separate idea.
COMPACT = ["roster_obpm", "roster_dbpm", "star_power", "roster_age", "continuity",
           "prev_net_x_cont", "opp3_luck"]
FEATURE_SETS = {"weighted": FEATURES, "compact": COMPACT, "bpm_only": BPM_ONLY}
ROLL_SEASONS = range(2015, 2026)           # rolling predictions: tune on 2015-2020, test on 2021-2025
TUNE_SEASONS = range(2015, 2021)
BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]  # weight on the statistics model in a blend with the combined model

DESCRIPTIONS = {
    "roster_obpm": "Roster offensive BPM", "roster_dbpm": "Roster defensive BPM",
    "roster_ws48": "Roster win shares per 48", "roster_ts": "Roster true shooting %",
    "roster_tov": "Roster turnover %", "roster_ast": "Roster assist %", "roster_orb": "Roster offensive rebound %",
    "roster_drb": "Roster defensive rebound %", "roster_stl": "Roster steal %", "roster_blk": "Roster block %",
    "star_power": "Top-2 player BPM", "bench_depth": "6th-10th man BPM", "roster_age": "Roster age",
    "prev_net_x_cont": "Last season net rating x continuity", "efg_diff_x_cont": "Last season shooting edge x continuity",
    "tov_diff_x_cont": "Last season turnover edge x continuity", "oreb_diff_x_cont": "Last season rebounding edge x continuity",
    "ftr_diff_x_cont": "Last season free-throw edge x continuity", "continuity": "Returning minutes share",
    "opp3_luck": "Opponent 3P% luck last season",
}


# ---------------------------------------------------------------- data
def fetch_br_full(end_year):
    """Basketball-Reference advanced stats with every column we use, cached."""
    path = os.path.join(DATA_DIR, f"br_advanced_full_{end_year}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    import requests

    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_advanced.html"
    print(f"Basketball-Reference (all advanced columns) {end_year - 1}-{str(end_year)[-2:]}")
    for _ in range(4):
        time.sleep(4)
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (personal research project)"}, timeout=30)
        if r.status_code == 200:
            break
        print(f"  status {r.status_code}, retrying in 60s")
        time.sleep(60)
    r.raise_for_status()
    r.encoding = "utf-8"
    tables = pd.read_html(StringIO(r.text))
    df = next(t for t in tables if "BPM" in [c[-1] if isinstance(c, tuple) else c for c in t.columns])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]
    df = df.rename(columns={"Tm": "Team"})
    df = df[(df["Player"] != "Player") & (df["Player"] != "League Average")].copy()
    df = df[[c for c in BR_KEEP if c in df.columns]].dropna(subset=["Player", "Team"])
    for c in df.columns:
        if c not in ("Player", "Team"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Player"] = df["Player"].str.replace("*", "", regex=False).str.strip()
    df["row"] = range(len(df))
    df.to_csv(path, index=False)
    return df


def player_seasons_full():
    rows = []
    for end_year in range(B.BR_FIRST_END_YEAR, LAST_COMPLETED + 2):
        df = fetch_br_full(end_year).sort_values("row")
        for player, g in df.groupby("Player", sort=False):
            totals = g[g["Team"].map(B.is_total_row)]
            teams = g[~g["Team"].map(B.is_total_row)]
            if teams.empty:
                continue
            tot = totals.iloc[0] if not totals.empty else teams.iloc[0]
            row = {"key": B.name_key(player), "player": player, "season": end_year - 1,
                   "age": tot["Age"], "G": tot["G"], "MP": tot["MP"], "first_team": teams.iloc[0]["Team"]}
            for s in RATE_STATS + list(IMPACT_PRIORS):
                row[s] = tot.get(s, np.nan)
            rows.append(row)
    out = pd.DataFrame(rows).drop_duplicates(["key", "season"])
    out["TEAM_ID"] = out["first_team"].map(B.BR_TEAM_IDS)
    return out


def fetch_team_factors():
    """Four Factors and opponent 3P% for every team-season from NBA.com, cached."""
    path = os.path.join(DATA_DIR, "team_four_factors.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    from nba_api.stats.endpoints import leaguedashteamstats

    rows = []
    for y in range(2010, LAST_COMPLETED + 1):
        print(f"team Four Factors {B.season_str(y)}")
        ff = None
        for measure in ["Four Factors", "Opponent"]:
            for attempt in range(4):
                try:
                    time.sleep(0.8)
                    df = leaguedashteamstats.LeagueDashTeamStats(
                        season=B.season_str(y), measure_type_detailed_defense=measure, timeout=60).get_data_frames()[0]
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"Couldn't get {measure} stats for {B.season_str(y)}")
            keep = (["TEAM_ID", "EFG_PCT", "FTA_RATE", "TM_TOV_PCT", "OREB_PCT", "OPP_EFG_PCT",
                     "OPP_FTA_RATE", "OPP_TOV_PCT", "OPP_OREB_PCT"] if measure == "Four Factors"
                    else ["TEAM_ID", "OPP_FG3_PCT"])
            df = df[keep]
            ff = df if ff is None else ff.merge(df, on="TEAM_ID")
        rows.append(ff.assign(season=y))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


# ----------------------------------------------------------- projections
def aging_curves(ps, last_season):
    curves = {}
    for stat in IMPACT_PRIORS:
        d = ps[["key", "season", "age", stat, "MP"]].rename(columns={stat: "BPM"}).dropna()
        curves[stat] = B.aging_curve(d, last_season)
    return curves


def project(ps, target, curves):
    """Projected per-player stats and raw minutes going into `target`."""
    hist = ps[(ps.season >= target - 3) & (ps.season <= target - 1)].copy()
    hist["sw"] = hist.season.map({target - 1: B.SEASON_WEIGHTS[0], target - 2: B.SEASON_WEIGHTS[1],
                                  target - 3: B.SEASON_WEIGHTS[2]})
    hist["w"] = hist.sw / hist.groupby("key").sw.transform("sum") * hist.MP
    last = ps[ps.season == target - 1]
    league = {s: np.average(last[s].fillna(last[s].mean()), weights=last.MP) for s in RATE_STATS}

    g = hist.groupby("key")
    out = pd.DataFrame({"last_age": g.age.last(), "last_season": g.season.max()})
    den = g.w.sum()
    for s in RATE_STATS + list(IMPACT_PRIORS):
        prior = IMPACT_PRIORS.get(s, league.get(s))
        num = (hist[s].fillna(prior) * hist.w).groupby(hist.key).sum()
        out[s] = (num + PRIOR_MINUTES * prior) / (den + PRIOR_MINUTES)
        if s in curves:
            out[s] += out.last_age.round().clip(18, 41).map(curves[s]).fillna(0)
    out["age_next"] = out.last_age + (target - out.last_season)

    recent = hist[(hist.season >= target - 2) & (hist.G >= B.MIN_GAMES_FOR_MPG)].sort_values("season")
    mpg = recent.groupby("key").apply(lambda x: x.MP.iloc[-1] / x.G.iloc[-1], include_groups=False)
    out["raw_min"] = (mpg * B.EXPECTED_GAMES).reindex(out.index).fillna(300).clip(upper=B.MAX_MINUTES)
    return out, league


def roster_features(roster, proj, league, rookie_age=21):
    """roster: [key, TEAM_ID]. Returns one row of roster features per team."""
    r = roster.merge(proj, left_on="key", right_index=True, how="left")
    for s, v in ROOKIE_IMPACT.items():
        r[s] = r[s].fillna(v)
    for s in RATE_STATS:
        r[s] = r[s].fillna(league[s])
    r["age_next"] = r.age_next.fillna(rookie_age)
    r["raw_min"] = r.raw_min.fillna(B.ROOKIE_MINUTES)
    r["proj_min"] = r.groupby("TEAM_ID").raw_min.transform(B.allocate_minutes)
    r["w"] = r.proj_min / B.TEAM_MINUTES

    def team(g):
        w = g.w
        top = g[g.proj_min >= 1200].nlargest(2, "BPM")
        if len(top) < 2:
            top = g.nlargest(2, "proj_min")
        bench = g.nlargest(10, "proj_min").iloc[5:10]
        return pd.Series({
            "roster_bpm": (w * g.BPM).sum() * 5,
            "roster_obpm": (w * g.OBPM).sum() * 5,
            "roster_dbpm": (w * g.DBPM).sum() * 5,
            "roster_ws48": (w * g["WS/48"]).sum(),
            "roster_ts": (w * g["TS%"]).sum() - league["TS%"],
            "roster_tov": (w * g["TOV%"]).sum(),
            "roster_ast": (w * g["AST%"]).sum(),
            "roster_orb": (w * g["ORB%"]).sum(),
            "roster_drb": (w * g["DRB%"]).sum(),
            "roster_stl": (w * g["STL%"]).sum(),
            "roster_blk": (w * g["BLK%"]).sum(),
            "star_power": top.BPM.mean(),
            "bench_depth": np.average(bench.BPM, weights=bench.proj_min.clip(lower=1)) if len(bench) else 0,
            "roster_age": (w * g.age_next).sum(),
        })
    return r.groupby("TEAM_ID").apply(team, include_groups=False), r


def team_features(ps, factors, teams, roster, target):
    """Last season's team stats, discounted by how much of the roster is back."""
    prev = ps[ps.season == target - 1][["key", "TEAM_ID", "MP"]].dropna()
    back = prev.merge(roster[["key", "TEAM_ID"]].rename(columns={"TEAM_ID": "NEXT"}), on="key", how="left")
    cont = (back[back.TEAM_ID == back.NEXT].groupby("TEAM_ID").MP.sum()
            / prev.groupby("TEAM_ID").MP.sum()).rename("continuity")

    f = factors[factors.season == target - 1].set_index("TEAM_ID")
    t = teams[teams.season == target - 1].set_index("TEAM_ID")
    out = pd.DataFrame(index=t.index)
    out["continuity"] = cont.reindex(out.index).fillna(0)
    c = out.continuity
    out["prev_net_x_cont"] = t.NET_RATING * c
    out["efg_diff_x_cont"] = (f.EFG_PCT - f.OPP_EFG_PCT) * c
    out["tov_diff_x_cont"] = (f.OPP_TOV_PCT - f.TM_TOV_PCT) * c
    out["oreb_diff_x_cont"] = (f.OREB_PCT - f.OPP_OREB_PCT) * c
    out["ftr_diff_x_cont"] = (f.FTA_RATE - f.OPP_FTA_RATE) * c
    out["opp3_luck"] = f.OPP_FG3_PCT.mean() - f.OPP_FG3_PCT
    out["prev_net"] = t.NET_RATING
    out["prev_wins_82"] = t.W_PCT * 82
    return out


def build_rows(ps, factors, teams, target, roster, curves):
    proj, league = project(ps, target, curves)
    rf, players = roster_features(roster, proj, league)
    tf = team_features(ps, factors, teams, roster, target)
    return rf.join(tf, how="inner"), players


def build_history(ps, factors, teams):
    rows = []
    for target in range(FIRST_TARGET, LAST_COMPLETED + 1):
        curves = aging_curves(ps, target - 1)
        roster = ps.loc[ps.season == target, ["key", "TEAM_ID"]].dropna()
        df, _ = build_rows(ps, factors, teams, target, roster, curves)
        t = teams[teams.season == target].set_index("TEAM_ID")[["TEAM_NAME", "NET_RATING", "W_PCT"]]
        df = df.join(t, how="inner").assign(season=target)
        rows.append(df.reset_index())
    h = pd.concat(rows, ignore_index=True)
    h["wins_82"] = h.W_PCT * 82
    return h


# ------------------------------------------------------------- modeling
def fit_ridge(train, feats):
    """Standardize, pick the ridge penalty by leave-one-season-out CV, then fit on all of train."""
    scaler = StandardScaler().fit(train[feats])
    X = scaler.transform(train[feats])
    y = train.NET_RATING.to_numpy()
    seasons = train.season.to_numpy()
    best, best_err = None, np.inf
    for a in ALPHAS:
        errs = []
        for s in np.unique(seasons):
            m = Ridge(alpha=a).fit(X[seasons != s], y[seasons != s])
            errs.append(np.mean((m.predict(X[seasons == s]) - y[seasons == s]) ** 2))
        if np.mean(errs) < best_err:
            best, best_err = a, np.mean(errs)
    return scaler, Ridge(alpha=best).fit(X, y), best


def to_wins(train_pred, train_wins, pred):
    """Convert predicted net rating to wins with a line fit on the training seasons."""
    lr = LinearRegression().fit(train_pred.reshape(-1, 1), train_wins)
    w = lr.predict(pred.reshape(-1, 1))
    return w + (41 * len(w) - w.sum()) / len(w)


def chart(rank, path):
    r = rank.sort_values("power_points")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(r.TEAM_NAME, r.power_points, color=np.where(r.power_points >= 0, "#3b6ea5", "#c0504d"))
    ax.axvline(0, color="#888", lw=1)
    for i, (v, rk) in enumerate(zip(r.power_points, r["rank"])):
        ax.text(v + (0.15 if v >= 0 else -0.15), i, f"#{rk}  {v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Power points (points per game vs. an average team, neutral court)")
    ax.set_title(f"{B.season_str(TARGET)} preseason power rankings (weighted statistics)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def redundancy_report(hist):
    """Pairs of features that move together closely enough to carry mostly the same information."""
    c = hist[FEATURES].corr()
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]]
    rep = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation"])
    rep = rep.reindex(rep.correlation.abs().sort_values(ascending=False).index)
    rep.round(2).to_csv(os.path.join(OUT_DIR, "redundancy_report.csv"), index=False)
    return rep


def rolling_stats_model(hist, feats):
    """Each season predicted from earlier seasons only, converted to wins."""
    out = []
    for s in ROLL_SEASONS:
        tr, te = hist[hist.season < s], hist[hist.season == s]
        sc, m, _ = fit_ridge(tr, feats)
        p_tr, p_te = m.predict(sc.transform(tr[feats])), m.predict(sc.transform(te[feats]))
        out.append(pd.DataFrame({"TEAM_ID": te.TEAM_ID.to_numpy(), "season": s,
                                 "wins": to_wins(p_tr, tr.wins_82.to_numpy(), p_te)}))
    return pd.concat(out, ignore_index=True)


def rolling_combined(teams):
    """The existing combined model (Version A + Version B) predicted the same rolling way."""
    import nba_wins_version_a as A
    import nba_wins_ensemble as E
    hist_a = A.build_history(A.fetch_team_seasons(), A.fetch_player_seasons())
    pred_a = E.rolling_predictions(hist_a, E.FEATS_A, ROLL_SEASONS, "pred_A")
    ps_b = B.player_seasons_br()
    curves = {t: B.aging_curve(ps_b, t - 1) for t in range(B.FIRST_TARGET, B.TARGET + 1)}
    hist_b = B.build_history(ps_b, teams, PRIOR_MINUTES, curves)
    pred_b = E.rolling_predictions(hist_b, E.FEATS_B, ROLL_SEASONS, "pred_B")
    df = pred_a.merge(pred_b[["TEAM_ID", "season", "pred_B"]], on=["TEAM_ID", "season"])
    tune = df[df.season.isin(TUNE_SEASONS)]
    w = min(E.WEIGHT_OPTIONS, key=lambda x: mean_absolute_error(
        tune.wins_82, (1 - x) * tune.pred_A + x * tune.pred_B))
    df["wins"] = (1 - w) * df.pred_A + w * df.pred_B
    return df[["TEAM_ID", "season", "wins_82", "prev_wins_82", "wins"]], w


def compare_options(hist, teams):
    combined, w_ab = rolling_combined(teams)
    preds = combined.rename(columns={"wins": "combined"})
    for name, feats in FEATURE_SETS.items():
        p = rolling_stats_model(hist, feats).rename(columns={"wins": name})
        preds = preds.merge(p, on=["TEAM_ID", "season"])

    # blends with the combined model; the weight is picked on the tuning seasons only
    blend_w = {}
    tune = preds[preds.season.isin(TUNE_SEASONS)]
    for name in ["weighted", "compact"]:
        w = min(BLEND_WEIGHTS, key=lambda x: mean_absolute_error(
            tune.wins_82, (1 - x) * tune.combined + x * tune[name]))
        blend_w[name] = w
        preds[f"combined_{name}_50_50"] = 0.5 * preds.combined + 0.5 * preds[name]
        preds[f"combined_{name}_tuned"] = (1 - w) * preds.combined + w * preds[name]

    options = ["combined", "weighted", "compact", "bpm_only", "combined_weighted_50_50", "combined_weighted_tuned",
               "combined_compact_50_50", "combined_compact_tuned"]
    rows = []
    for s, g in preds[preds.season.isin(TEST_SEASONS)].groupby("season"):
        row = {"season": B.season_str(s)}
        row.update({o: mean_absolute_error(g.wins_82, g[o]) for o in options})
        row["repeat_last_year"] = mean_absolute_error(g.wins_82, g.prev_wins_82)
        rows.append(row)
    bt = pd.DataFrame(rows)
    tune_err = {o: mean_absolute_error(tune.wins_82, tune[o]) if o in tune else
                mean_absolute_error(preds[preds.season.isin(TUNE_SEASONS)].wins_82,
                                    preds[preds.season.isin(TUNE_SEASONS)][o]) for o in options}
    return bt, tune_err, blend_w, w_ab


def final_stats_wins(hist, cur, feats):
    """Fit on every season and predict 2026-27 wins."""
    sc, m, alpha = fit_ridge(hist, feats)
    p_hist = m.predict(sc.transform(hist[feats]))
    z = sc.transform(cur[feats])
    return to_wins(p_hist, hist.wins_82.to_numpy(), m.predict(z)), sc, m, z, alpha


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    teams = pd.read_csv(os.path.join(DATA_DIR, "team_seasons.csv"))
    ps = player_seasons_full()
    factors = fetch_team_factors()
    hist = build_history(ps, factors, teams)
    hist.to_csv(os.path.join(OUT_DIR, "training_data.csv"), index=False)

    # 1. redundancy
    rep = redundancy_report(hist)
    print("\nMost redundant pairs of statistics (correlation; 1.0 = identical information):")
    print(rep.head(10).round(2).to_string(index=False))

    # 2. compare every option on the same seasons
    print("\nBuilding rolling predictions for every option (2015-16 to 2025-26)...")
    bt, tune_err, blend_w, w_ab = compare_options(hist, teams)
    bt.to_csv(os.path.join(OUT_DIR, "backtest_options.csv"), index=False)
    print(f"\nBlend weights picked on 2015-16 to 2020-21 (weight on the statistics model): "
          f"weighted {int(blend_w['weighted'] * 100)}%, compact {int(blend_w['compact'] * 100)}%")
    print("\nTest seasons, average error in wins per team (lower is better):")
    print(bt.round(2).to_string(index=False))
    means = bt.drop(columns="season").mean().sort_values()
    print("\nAverage, best first:")
    print(means.round(2).to_string())

    best = min(tune_err, key=tune_err.get)   # chosen on the tuning seasons, not the test seasons
    print(f"\nChosen for the preseason power rankings: {best} "
          f"(lowest error on 2015-16 to 2020-21; test-season error {means[best]:.2f})")

    # 3. weights for both statistics models
    team_ids = teams.loc[teams.season == LAST_COMPLETED, "TEAM_ID"].unique()
    rosters = B.fetch_target_rosters_named(team_ids)
    rosters["key"] = rosters.player.map(B.name_key)
    cur, _ = build_rows(ps, factors, teams, TARGET, rosters[["key", "player", "TEAM_ID"]],
                        aging_curves(ps, LAST_COMPLETED))
    wins = {}
    for name, feats in [("weighted", FEATURES), ("compact", COMPACT)]:
        wins[name], sc, m, z, alpha = final_stats_wins(hist, cur, feats)
        wt = pd.DataFrame({"feature": feats, "description": [DESCRIPTIONS[f] for f in feats], "weight": m.coef_})
        wt["share_of_importance_pct"] = wt.weight.abs() / wt.weight.abs().sum() * 100
        wt = wt.sort_values("share_of_importance_pct", ascending=False).round(2)
        wt.to_csv(os.path.join(OUT_DIR, f"feature_weights_{name}.csv"), index=False)
        print(f"\nLearned weights, {name} set (points of net rating per 1 standard deviation; penalty {alpha:.1f}):")
        print(wt.to_string(index=False))
        if name == "compact":
            contrib = pd.DataFrame(z * m.coef_, index=cur.index, columns=feats)
            contrib = contrib - contrib.mean()

    # 4. 2026-27 under every option, then the chosen one
    names = teams[teams.season == LAST_COMPLETED].set_index("TEAM_ID").TEAM_NAME
    ens = pd.read_csv(os.path.join("output_ensemble", f"projections_ensemble_{B.season_str(TARGET)}.csv"),
                      encoding="utf-8-sig")
    ens_col = [c for c in ens.columns if c.startswith("tuned")][0]
    out = pd.DataFrame({"TEAM_NAME": names})
    out["combined"] = out.TEAM_NAME.map(ens.set_index("TEAM_NAME")[ens_col])
    out["weighted"] = pd.Series(wins["weighted"], index=cur.index)
    out["compact"] = pd.Series(wins["compact"], index=cur.index)
    for name in ["weighted", "compact"]:
        out[f"combined_{name}_50_50"] = 0.5 * out.combined + 0.5 * out[name]
        out[f"combined_{name}_tuned"] = (1 - blend_w[name]) * out.combined + blend_w[name] * out[name]
    if best == "bpm_only":
        best = "combined"  # bpm_only is a reference point, not a full model
    out["projected_wins"] = out[best]
    out["power_points"] = (out.projected_wins - 41) / 2.7
    out["power_points"] -= out.power_points.mean()
    out["conf"] = np.where(out.TEAM_NAME.isin(EAST), "East", "West")
    out = out.sort_values("power_points", ascending=False).reset_index()
    out.insert(0, "rank", out.index + 1)
    out["model_used"] = best
    factors_txt = []
    for tid in out.TEAM_ID:
        c = contrib.loc[tid].sort_values()
        factors_txt.append(", ".join(f"{DESCRIPTIONS[k]} {v:+.1f}" for k, v in
                                     list(c.iloc[::-1].head(2).items()) + list(c.head(1).items())))
    out["biggest_factors"] = factors_txt
    cols = ["rank", "TEAM_NAME", "conf", "power_points", "projected_wins", "model_used", "biggest_factors"]
    out[cols].round(1).to_csv(os.path.join(OUT_DIR, f"preseason_power_{B.season_str(TARGET)}.csv"),
                              index=False, encoding="utf-8-sig")
    opt_cols = ["TEAM_NAME", "combined", "weighted", "compact", "combined_weighted_tuned", "combined_compact_tuned"]
    out[opt_cols].round(1).to_csv(os.path.join(OUT_DIR, f"all_options_{B.season_str(TARGET)}.csv"),
                                  index=False, encoding="utf-8-sig")
    contrib.assign(TEAM_NAME=names).round(2).to_csv(os.path.join(OUT_DIR, "feature_contributions.csv"),
                                                    encoding="utf-8-sig")

    print(f"\n{B.season_str(TARGET)} preseason power rankings ({best}):")
    print(out[cols[:5]].round(1).to_string(index=False))
    print("\nProjected wins under each option:")
    print(out[opt_cols].round(1).to_string(index=False))
    out = out.rename(columns={"power_points": "power_points"})
    chart(out, os.path.join(OUT_DIR, f"preseason_power_{B.season_str(TARGET)}.png"))
    print(f"\nSaved results to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
