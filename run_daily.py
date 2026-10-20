"""
Daily routine

Runs the in-season scripts in order and saves everything they print to logs/<date>.log.

  Full run (late morning):  power ratings -> season sim -> player props -> grading -> Google Sheet sync -> website
  Props run (afternoon):    player props -> Google Sheet sync -> website, to pick up the latest injury report

If the power ratings fail, the run stops, since everything after them depends on that day's ratings.
Any other failure is logged and the run continues. A day with no games is normal: the props step
says so and moves on. The Google Sheet sync is skipped until google_service_account.json is set up.

Close NBA_Spread_Calculator.xlsx in Excel before a run.

Usage:
  python3 run_daily.py           # full run
  python3 run_daily.py --props   # props and sync only
"""

import datetime as dt
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FULL = [("Power ratings", ["nba_power_ratings.py"], True),
        ("Season simulation", ["nba_season_sim.py"], False),
        ("Player projections", ["nba_player_props.py"], False),
        ("Grading", ["nba_grade.py"], False),
        ("Google Sheet sync", ["sync_to_sheets.py"], False),
        ("Website", ["build_site.py", "--publish"], False)]
PROPS = [("Player projections", ["nba_player_props.py"], False),
         ("Google Sheet sync", ["sync_to_sheets.py"], False),
         ("Website", ["build_site.py", "--publish"], False)]


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    steps = PROPS if "--props" in sys.argv else FULL
    started = dt.datetime.now()
    log_path = os.path.join(LOG_DIR, f"{started:%Y-%m-%d}.log")
    summary = []
    with open(log_path, "a", encoding="utf-8") as log:
        def out(text):
            print(text)
            log.write(text + "\n")
            log.flush()

        out(f"\n{'=' * 70}\n{'Props run' if steps is PROPS else 'Full run'} started {started:%Y-%m-%d %H:%M}")
        for name, args, required in steps:
            if name == "Google Sheet sync" and not os.path.exists(os.path.join(HERE, "google_service_account.json")):
                out(f"\n--- {name}: skipped (google_service_account.json not set up yet)")
                summary.append((name, "skipped"))
                continue
            out(f"\n--- {name}")
            p = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True, text=True)
            text = (p.stdout + p.stderr).rstrip()
            out(text)
            if p.returncode == 0:
                status = "ok"
            elif "No games on" in text:
                status = "no games today"
            else:
                status = f"FAILED (exit code {p.returncode})"
            summary.append((name, status))
            if required and status.startswith("FAILED"):
                out(f"\n{name} failed, so the rest of the run was skipped.")
                break
        secs = int((dt.datetime.now() - started).total_seconds())
        out(f"\nSummary ({secs // 60} min {secs % 60} sec):")
        for name, status in summary:
            out(f"  {name:<20} {status}")
    if any(s.startswith("FAILED") for _, s in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
