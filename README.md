# 2026-27-NBA-Win-Projections
Team-level regression that projects NBA win totals for 2026-27, built with Python, nba_api and scikit-learn and backtested over five seasons.

NBA Win Projections: 2026-27

A regression model that projects regular-season wins for all 30 NBA teams, built from team and player data going back to 2010-11. This is Version A, a deliberately simple team-level model. It gives a baseline to measure a more detailed player-level version against.

![2026-27 projected wins](output/projections_2026-27.png)

## Why this approach

A team's point differential says more about how good it is than its record does, and it holds up better from one season to the next. So the model starts from last season's net rating (points scored minus points allowed per 100 possessions). It then adjusts for two things that change over an offseason: how much of the roster is still around, and how old it is.

Every input is known before opening night. The model never sees anything from the season it's predicting, apart from who is on each roster.

## Features

| Feature | Description |
|---|---|
| `prev_net_rating` | Last season's net rating |
| `returning_min_share` | Share of last season's minutes played by players still on the roster |
| `net_x_returning` | Net rating multiplied by returning share. Last year's rating should count for more when the roster stayed together |
| `roster_age` | Average age of the upcoming roster, weighted by last season's minutes |

The target is next season's wins, scaled to 82 games so the shortened seasons (2011-12, 2019-20, 2020-21) line up with the rest. Projections are shifted so the league total equals 1,230 wins, since every game has exactly one winner.

## Results

I backtested the model on the five most recent seasons. For each season, the model was trained only on the seasons before it, the same way it would have been used in real time. It was compared against two baselines: every team wins 41, and every team repeats last season's win total.

**Mean absolute error, in wins per team:**

| Season | Model | Repeat last season | Every team wins 41 |
|---|---|---|---|
| 2021-22 | 7.0 | 8.2 | 9.7 |
| 2022-23 | 6.4 | 8.1 | 7.5 |
| 2023-24 | 7.9 | 8.5 | 11.2 |
| 2024-25 | 7.5 | 9.1 | 10.6 |
| 2025-26 | 9.1 | 10.4 | 11.6 |
| **Average** | **7.6** | **8.8** | **10.1** |

The model beat both baselines in all five seasons. On average it cut error by about 14% against "repeat last season" and about 25% against "every team wins 41." Published projection systems and preseason betting lines usually land somewhere around 6 to 7 wins of error, so there's still room to close.

2025-26 was the hardest season to call for every method, which fits a year with a lot of injuries and teams openly playing for draft position.

### Fitted model

```
wins = -4.1 + 0.90 × prev_net_rating + 22.89 × returning_min_share
            + 0.41 × net_x_returning + 1.13 × roster_age
```

A few things stand out:

- For a team that brings back about 70% of its minutes, each point of net rating is worth roughly 1.2 wins the following season. Within a single season a point is worth about 2.7 wins, so a little under half of a team's edge carries over and the rest fades toward average.
- The interaction term is positive. Last year's rating is a better guide for teams that kept their roster together.
- Returning minutes adds about 2.3 wins for every 10 percentage points. Some of that is continuity, but some is probably selection: good teams keep their players, bad teams get rebuilt.
- Older rosters project slightly better. That's almost certainly because contenders give minutes to proven veterans, not because aging helps. See below.

### What I tested and dropped

Since average age came out with a positive sign, I tried replacing it with the share of minutes going to players 31 and older, which should pick up aging risk more directly. It performed slightly worse (7.68 average error vs. 7.60) and still came out positive, at about +0.6 wins per 10 percentage points. At the team level, age mostly works as a stand-in for roster quality. Real aging effects will be handled at the player level in Version B. The script still runs both versions and reports them side by side.

## 2026-27 projections

Made on September 30, 2026, before the start of the regular season.

| Team | Projected wins | 2025-26 wins |
|---|---|---|
| San Antonio Spurs | 56.5 | 62 |
| New York Knicks | 55.5 | 53 |
| Boston Celtics | 53.6 | 56 |
| Oklahoma City Thunder | 53.3 | 64 |
| Detroit Pistons | 51.6 | 60 |
| Houston Rockets | 51.0 | 52 |
| Denver Nuggets | 47.6 | 54 |
| Cleveland Cavaliers | 46.7 | 52 |
| Orlando Magic | 46.3 | 45 |
| Golden State Warriors | 46.0 | 37 |
| Charlotte Hornets | 44.9 | 44 |
| Atlanta Hawks | 44.5 | 46 |
| Miami Heat | 44.0 | 43 |
| Phoenix Suns | 43.7 | 45 |
| Toronto Raptors | 43.5 | 46 |
| Minnesota Timberwolves | 42.7 | 49 |
| LA Clippers | 40.8 | 42 |
| Portland Trail Blazers | 40.0 | 42 |
| Philadelphia 76ers | 37.2 | 45 |
| New Orleans Pelicans | 36.8 | 26 |
| Los Angeles Lakers | 35.1 | 53 |
| Indiana Pacers | 35.0 | 19 |
| Milwaukee Bucks | 32.8 | 32 |
| Chicago Bulls | 31.4 | 31 |
| Memphis Grizzlies | 30.8 | 25 |
| Dallas Mavericks | 30.6 | 26 |
| Utah Jazz | 28.3 | 22 |
| Sacramento Kings | 27.7 | 22 |
| Brooklyn Nets | 26.1 | 20 |
| Washington Wizards | 25.7 | 17 |

Some notes on the list:

- **Regression to the mean does most of the work at both ends.** Oklahoma City drops 11 wins and Washington gains 9, mostly because extreme seasons rarely repeat.
- **Golden State** won 37 games with a net rating near zero, which is closer to a .500 team. The model trusts the point differential over the record.
- **The Lakers** brought back only 42% of last season's minutes, the lowest in the league, and the model takes them down 18 wins. It knows how much of the roster left but not how good the replacements are, so this is the projection I trust least.
- **Indiana** is probably too low. Most of their 19-win season came without Tyrese Haliburton, and the model has no way of knowing he's expected back.

## Limitations

- **Injuries aren't modeled.** Teams getting a star back from injury will be underrated, and teams relying on injury-prone players will be overrated.
- **New players are only counted as a share of minutes.** A team that loses a starter and signs an All-Star looks the same as one that loses a starter and signs no one.
- **Projections are compressed.** They run from about 26 to 57 wins, while actual seasons usually range from about 15 to 65. Pulling toward the middle lowers average error but misses the true outliers.
- **Traded players** are assigned to the team they finished the season with, following NBA.com's convention.

## Next steps

- Record preseason sportsbook win totals and compare them with these projections once the season ends.
- **Version B:** build team ratings up from individual players, using a weighted three-year plus-minus rating for each player, a proper aging curve and projected minutes. This should address the injury and roster-quality blind spots above.
- **Version C:** simulate the full schedule several thousand times to get a range and playoff odds for each team, not just a single number.

## Running it

```
pip install nba_api pandas scikit-learn matplotlib
python3 nba_wins_version_a.py
```

The first run pulls data from NBA.com through the `nba_api` package and takes about five minutes, since it pauses between requests. Data is cached in `./data`, so later runs finish in seconds. To pick up roster moves before the season starts, delete `data/rosters_2026.csv` and run it again.

**Output files (`./output`):**
- `backtest.csv`: season-by-season error for both age versions and both baselines
- `projections_2026-27.csv` / `.png`: the projections above
- `training_data.csv`: the full modeling table, one row per team-season

## Data

Team and player statistics come from NBA.com via [`nba_api`](https://github.com/swar/nba_api). Coverage runs from the 2010-11 season through 2025-26, with 2026-27 rosters as of the date above.
