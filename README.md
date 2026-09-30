NBA Win Projections: 2026-27

LIVE DASHBOARD: https://public.tableau.com/app/profile/jt.mcgahran/viz/026-27NBAProjectionsupdated9_30_26/26-27WinProjections


Preseason win projections for all 30 NBA teams, built from two models and a weighted average of both:

- **Version A** works at the team level. It starts from last season's point differential and adjusts for roster turnover and age.
- **Version B** works at the player level. It projects every player's rating and minutes, then adds them up into a team rating.
- **The combined model** weights B at 75% and A at 25%. It had the lowest error in backtesting: **6.95 wins per team, 21% better than assuming every team repeats last season.**


## Results

Each model was backtested on the five most recent seasons. For every season, the model was trained only on the seasons before it, the same way it would have been used in real time. The combined model's 75/25 weighting was chosen using 2015-16 through 2020-21, so the test seasons played no part in picking it.

**Mean absolute error, in wins per team:**

| Season | Combined (75% B) | 50/50 average | Version B | Version A | Repeat last season | Every team wins 41 |
|---|---|---|---|---|---|---|
| 2021-22 | 7.1 | 7.1 | 7.8 | 7.0 | 8.2 | 9.7 |
| 2022-23 | 4.4 | 4.9 | 4.7 | 6.4 | 8.1 | 7.5 |
| 2023-24 | 7.8 | 7.7 | 8.0 | 7.9 | 8.5 | 11.2 |
| 2024-25 | 7.2 | 7.2 | 7.2 | 7.5 | 9.1 | 10.6 |
| 2025-26 | 8.2 | 8.4 | 8.2 | 9.1 | 10.4 | 11.6 |
| **Average** | **6.95** | **7.08** | **7.17** | **7.60** | **8.84** | **10.12** |

The combined model beat or tied Version B in all five seasons and beat Version A in four. Published projection systems and preseason betting lines usually land somewhere around 6 to 7 wins of error, so this is in the same general range, though I haven't measured the market directly yet (see Next steps).

The gain from combining is modest. A and B's errors had a correlation of 0.83, which means they mostly miss on the same teams. Averaging still helps where they disagree, and B gets most of the weight because it knows more about each roster.

2025-26 was the hardest season for every method, which fits a year with a lot of injuries and teams openly playing for draft position.

## Version A: team-level model

The idea is that point differential says more about a team than its record does, and it holds up better from one season to the next. So the model starts from last season's net rating (points scored minus points allowed per 100 possessions) and adjusts for how much of the roster changed.

| Feature | Description |
|---|---|
| `prev_net_rating` | Last season's net rating |
| `returning_min_share` | Share of last season's minutes played by players still on the roster |
| `net_x_returning` | Net rating × returning share. Last year's rating should count for more when the roster stayed together |
| `roster_age` | Average age of the upcoming roster, weighted by last season's minutes |

```
wins = -4.1 + 0.90 × prev_net_rating + 22.89 × returning_min_share
            + 0.41 × net_x_returning + 1.13 × roster_age
```

For a team bringing back about 70% of its minutes, each point of net rating is worth roughly 1.2 wins the next season. Within a single season a point is worth about 2.7 wins, so a little under half of a team's edge carries over and the rest fades toward average.

The age coefficient is positive, which looks backwards. It's almost certainly because contenders give minutes to proven veterans, so at the team level age acts as a proxy for quality. I also tested the share of minutes going to players 31 and older as a more direct aging measure. It did slightly worse (7.68 vs. 7.60) and still came out positive, so I kept average age and left real aging effects to Version B.

**Blind spots:** A knows how much of a roster left but not who replaced them, and it has no idea when an injured star is coming back.

## Version B: player-level model

Version B builds each team's rating from its individual players, using Box Plus/Minus (BPM) from Basketball-Reference as the player rating. BPM estimates a player's impact in points per 100 possessions.

1. **Player rating.** Each player's last three seasons are weighted 60/30/10 across the seasons he actually played, and also by minutes. Small samples are pulled toward -2.0, roughly a fringe rotation player. How strongly to pull was tuned on 2015-2020, and 500 minutes of evidence worked best, although results were nearly identical across settings.
2. **Aging.** An aging curve is estimated from the data: the average year-to-year change in BPM at each age, using only seasons before the one being projected.
3. **Minutes.** Minutes per game from the player's most recent season with 20+ games, times 68 games, capped at about 36 a game. Minutes are then filled by depth chart until the team reaches 19,680 (48 × 5 × 82), so the top of the rotation gets full minutes and the end of the bench gets whatever is left.
4. **Team rating** is the minutes-weighted player ratings × 5, which approximates projected net rating.
5. **Wins:**

```
wins = 45.5 + 2.45 × team_rating
```

A slope close to the real-world 2.7 wins per point suggests the team ratings are well calibrated. I also tested adding last season's team net rating to B. It barely helped (coefficient 0.39, and less than 0.1 wins of improvement), since the player ratings already capture most of what the team numbers know.

### Problems I found and fixed along the way

- **Accented names weren't matching.** Basketball-Reference pages are UTF-8, but they were being read with the wrong encoding, so "Jokić" came through garbled and never matched NBA.com's roster. 24 players, including Jokić, Dončić, Porziņģis and Şengün, were being treated as rookies, which put Denver at 36 wins. The script now repairs the encoding and normalizes names before matching.
- **Minutes were spread too thin.** Preseason rosters carry 20 or more players, and scaling everyone's minutes by the same factor gave Jalen Brunson about 24 minutes a game and gave camp invites several hundred minutes each. Switching to depth-chart allocation fixed it. Backtest error actually rose slightly (7.01 to 7.17), because the old approach was accidentally helping by pulling every team toward average. I kept the fix because the model now does what it claims to, and the team ratings came out better calibrated.
- **Injured players were being discounted twice.** A player who missed last season had his older seasons treated as thin evidence and pulled hard toward average. Weights are now spread across the seasons a player actually played.

## 2026-27 projections

Made on September 30, 2026, before the regular season.

| Team | Combined | Version A | Version B | 2025-26 wins |
|---|---|---|---|---|
| Oklahoma City Thunder | 58.7 | 53.3 | 60.5 | 64 |
| San Antonio Spurs | 53.6 | 56.5 | 52.6 | 62 |
| Houston Rockets | 52.6 | 51.0 | 53.1 | 52 |
| Boston Celtics | 52.5 | 53.6 | 52.2 | 56 |
| Denver Nuggets | 51.7 | 47.6 | 53.0 | 54 |
| Cleveland Cavaliers | 48.9 | 46.7 | 49.6 | 52 |
| New York Knicks | 47.6 | 55.5 | 45.0 | 53 |
| Philadelphia 76ers | 47.1 | 37.2 | 50.3 | 45 |
| Toronto Raptors | 46.9 | 43.5 | 48.1 | 46 |
| Detroit Pistons | 45.9 | 51.6 | 43.9 | 60 |
| Orlando Magic | 44.9 | 46.3 | 44.4 | 45 |
| Los Angeles Lakers | 44.7 | 35.1 | 47.9 | 53 |
| Golden State Warriors | 44.4 | 46.0 | 43.9 | 37 |
| Portland Trail Blazers | 43.4 | 40.0 | 44.5 | 42 |
| Charlotte Hornets | 43.0 | 44.9 | 42.4 | 44 |
| Minnesota Timberwolves | 42.2 | 42.7 | 42.1 | 49 |
| Miami Heat | 40.4 | 44.0 | 39.2 | 43 |
| Phoenix Suns | 40.1 | 43.7 | 38.9 | 45 |
| Atlanta Hawks | 39.7 | 44.5 | 38.1 | 46 |
| Dallas Mavericks | 38.8 | 30.6 | 41.5 | 26 |
| Indiana Pacers | 36.0 | 35.0 | 36.3 | 19 |
| New Orleans Pelicans | 34.9 | 36.8 | 34.3 | 26 |
| Chicago Bulls | 32.8 | 31.4 | 33.2 | 31 |
| LA Clippers | 31.9 | 40.8 | 29.0 | 42 |
| Milwaukee Bucks | 31.1 | 32.8 | 30.5 | 32 |
| Memphis Grizzlies | 30.3 | 30.8 | 30.1 | 25 |
| Utah Jazz | 29.9 | 28.3 | 30.4 | 22 |
| Brooklyn Nets | 26.4 | 26.1 | 26.4 | 20 |
| Washington Wizards | 26.1 | 25.7 | 26.3 | 17 |
| Sacramento Kings | 23.6 | 27.7 | 22.3 | 22 |

### Where the two models disagree

These are the teams to watch, since one of the two models is going to be clearly wrong:

- **Philadelphia (A 37, B 50).** B sees a talented roster when healthy. A only sees last season's roughly even point differential.
- **Lakers (A 35, B 48).** A sees 58% of last season's minutes gone and marks them down hard. B knows Luka Dončić is still there.
- **Clippers (A 41, B 29).** B rates the current roster as weak and older. A leans on last season's results.
- **Dallas (A 31, B 42).** B expects a real bounce-back from a 26-win season.
- **Knicks (A 56, B 45).** A likes their continuity and last season's +6.4 net rating. B rates the individual players more modestly.
- **Detroit (A 52, B 44).** B thinks last season's 60 wins ran ahead of the roster's talent.

## Limitations

- **Projections are compressed.** They run from about 24 to 59 wins, while actual seasons usually range from about 15 to 65. Pulling toward average lowers typical error but misses the true outliers.
- **BPM is built from box-score stats,** so it tends to undervalue players whose impact doesn't show up there, such as strong defenders and screeners.
- **Injuries during the season aren't modeled.** Players are assumed to play about 68 games.
- **The Version B backtest has a small head start.** Historical rosters are built from players who actually appeared for each team, so a player who missed an entire season is left out, which a real preseason projection wouldn't know. The backtest error is probably a few tenths of a win optimistic.
- **Traded players** are assigned to the team they finished the season with in Version A and the team they started with in Version B.

## Next steps

- Record preseason sportsbook win totals and compare both against actual results when the season ends.
- Simulate the full schedule a few thousand times to get a range and playoff odds for each team, not just a single number.
- Test a second player rating alongside BPM to see whether it helps with the defensive blind spot.

## Running it

All three scripts need to be in the same folder, and they share one `data` folder.

```
pip install -r requirements.txt
python3 nba_wins_version_a.py
python3 nba_wins_version_b.py
python3 nba_wins_ensemble.py
```

- **Version A** pulls team and player stats from NBA.com through `nba_api`. The first run takes about five minutes because of pauses between requests.
- **Version B** pulls one page per season from Basketball-Reference, 4 seconds apart to stay under their rate limit, plus 2026-27 rosters from NBA.com.
- **The combined model** uses the cached data from both and doesn't download anything.

Everything is cached in `./data`, so re-runs finish quickly. To pick up roster moves before opening night, delete `data/rosters_2026.csv` and `data/rosters_named_2026.csv` and run all three again.

**Output folders:**
- `output/`: Version A backtest, projections and chart
- `output_b/`: Version B backtest, team projections, player-by-player projections and chart
- `output_ensemble/`: side-by-side backtest of every model, final projections and the chart at the top of this page

## Data

- Team and player statistics: NBA.com via [`nba_api`](https://github.com/swar/nba_api), 2010-11 through 2025-26
- Player Box Plus/Minus: [Basketball-Reference](https://www.basketball-reference.com), 2007-08 through 2025-26
- 2026-27 rosters: NBA.com, as of the date above

Raw data files aren't included in this repository. The scripts download and cache them on the first run.
