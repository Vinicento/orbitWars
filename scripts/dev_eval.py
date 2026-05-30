"""
Parallel Orbit Wars evaluation.

Run from project root:
    python scripts/dev_eval.py

Behavior:
- every run creates a new folder: outputs/dev_eval/run_YYYYMMDD_HHMMSS/
- no overwritten tests
- no duplicated mirrored summary rows
- one row per matchup in matchup_summary.csv
- one row per matchup+seed in seed_results.csv
- multiprocessing for faster evaluation
- one combined plot
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Reduce optional Kaggle/OpenSpiel logging noise.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make  # noqa: E402
from owtools.metrics import final_rows_from_env, rows_from_env  # noqa: E402


# =====================
# EDIT ONLY THIS BLOCK
# =====================
ENV_NAME = "orbit_wars"
DEBUG = True

SEEDS = list(range(20))

MATCHUPS = [
    ["agents/passive.py", "agents/neutral_conquest.py"],
    ["agents/nearest_sniper.py", "agents/neutral_conquest.py"],
    ["agents/roi_expansion.py", "agents/neutral_conquest.py"],
    # ["main.py", "agents/neutral_conquest.py"],
]

OUT_ROOT = Path("outputs/dev_eval")

PARALLEL = True
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

SAVE_ONE_REPLAY_HTML = True
REPLAY_MATCHUP_INDEX = 0
REPLAY_SEED = SEEDS[0]
HTML_WIDTH = 1200
HTML_HEIGHT = 850

SAVE_PLAYER_RAW = False
SAVE_TURN_CURVES = False
# =====================


def make_run_dir(root: Path) -> Path:
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = root / run_id

    counter = 1
    while out_dir.exists():
        out_dir = root / f"{run_id}_{counter:02d}"
        counter += 1

    out_dir.mkdir(parents=True, exist_ok=False)
    (root / "latest_run.txt").write_text(str(out_dir.resolve()), encoding="utf-8")
    return out_dir


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def agent_label(agent_path: str) -> str:
    if agent_path in {"random", "reaction"}:
        return agent_path
    if agent_path == "main.py":
        return "main_submission"
    return Path(agent_path).with_suffix("").as_posix().replace("/", ".")


def matchup_label(agents: list[str]) -> str:
    return f"{agent_label(agents[0])} vs {agent_label(agents[1])}"


def print_config(out_dir: Path) -> None:
    print("=" * 90)
    print(f"ENV         : {ENV_NAME}")
    print(f"SEEDS       : {SEEDS}")
    print(f"PARALLEL    : {PARALLEL}")
    print(f"MAX_WORKERS : {MAX_WORKERS}")
    print(f"OUT_DIR     : {out_dir.resolve()}")
    print("MATCHUPS    :")
    for i, agents in enumerate(MATCHUPS):
        print(f"  {i}: {matchup_label(agents)} -> {agents}")
    print("=" * 90)


def save_config(out_dir: Path) -> None:
    config = {
        "env_name": ENV_NAME,
        "debug": DEBUG,
        "seeds": SEEDS,
        "matchups": MATCHUPS,
        "parallel": PARALLEL,
        "max_workers": MAX_WORKERS,
        "save_one_replay_html": SAVE_ONE_REPLAY_HTML,
        "replay_matchup_index": REPLAY_MATCHUP_INDEX,
        "replay_seed": REPLAY_SEED,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def one_seed_result(final: pd.DataFrame, agents: list[str], seed: int, matchup: str) -> dict:
    """
    Convert final player rows into one matchup+seed row.

    Assumes 2-player match:
    player 0 = agent_a
    player 1 = agent_b
    """
    a = final[final["player"] == 0].iloc[0]
    b = final[final["player"] == 1].iloc[0]

    agent_a = agent_label(agents[0])
    agent_b = agent_label(agents[1])

    a_rank = int(a["rank"])
    b_rank = int(b["rank"])

    if a_rank < b_rank:
        winner_side = "A"
        winner_agent = agent_a
    elif b_rank < a_rank:
        winner_side = "B"
        winner_agent = agent_b
    else:
        winner_side = "tie"
        winner_agent = "tie"

    return {
        "matchup": matchup,
        "seed": seed,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "winner_side": winner_side,
        "winner_agent": winner_agent,
        "a_rank": a_rank,
        "b_rank": b_rank,
        "a_reward": float(a["reward"]),
        "b_reward": float(b["reward"]),
        "a_total_ships": float(a["total_ships"]),
        "b_total_ships": float(b["total_ships"]),
        "a_planets": float(a["planet_count"]),
        "b_planets": float(b["planet_count"]),
        "a_production": float(a["production"]),
        "b_production": float(b["production"]),
    }


def run_one_job(job: tuple[int, list[str], int]) -> dict:
    """
    Worker-safe single simulation.

    Important:
    - no file writing here
    - no plotting here
    - returns data only
    """
    matchup_index, agents, seed = job
    matchup = matchup_label(agents)

    try:
        env = make(ENV_NAME, configuration={"seed": seed}, debug=DEBUG)
        env.run(agents)

        labels = dict(enumerate(agent_label(a) for a in agents))

        final = final_rows_from_env(env, seed)
        final["matchup_index"] = matchup_index
        final["matchup"] = matchup
        final["agent"] = final["player"].map(labels)

        turns = rows_from_env(env)
        turns["seed"] = seed
        turns["matchup_index"] = matchup_index
        turns["matchup"] = matchup
        turns["agent"] = turns["player"].map(labels)

        seed_row = one_seed_result(final, agents, seed, matchup)

        return {
            "ok": True,
            "matchup_index": matchup_index,
            "seed": seed,
            "agents": agents,
            "matchup": matchup,
            "seed_row": seed_row,
            "final": final,
            "turns": turns,
            "error_type": None,
            "error": None,
            "traceback": None,
        }

    except Exception as exc:
        return {
            "ok": False,
            "matchup_index": matchup_index,
            "seed": seed,
            "agents": agents,
            "matchup": matchup,
            "seed_row": None,
            "final": None,
            "turns": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def build_jobs() -> list[tuple[int, list[str], int]]:
    return [
        (matchup_index, agents, seed)
        for matchup_index, agents in enumerate(MATCHUPS)
        for seed in SEEDS
    ]


def summarize_matchups(seed_results: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    if seed_results.empty:
        return pd.DataFrame()

    rows = []

    for (matchup, agent_a, agent_b), sub in seed_results.groupby(["matchup", "agent_a", "agent_b"]):
        games_ok = len(sub)
        a_wins = int((sub["winner_side"] == "A").sum())
        b_wins = int((sub["winner_side"] == "B").sum())
        ties = int((sub["winner_side"] == "tie").sum())

        err_count = 0
        if not errors.empty:
            err_count = int((errors["matchup"] == matchup).sum())

        if a_wins > b_wins:
            best_agent = agent_a
        elif b_wins > a_wins:
            best_agent = agent_b
        else:
            best_agent = "tie"

        rows.append(
            {
                "matchup": matchup,
                "agent_a": agent_a,
                "agent_b": agent_b,
                "games_requested": len(SEEDS),
                "games_ok": games_ok,
                "errors": err_count,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "a_win_rate": round(a_wins / games_ok, 4) if games_ok else 0.0,
                "b_win_rate": round(b_wins / games_ok, 4) if games_ok else 0.0,
                "best_agent": best_agent,
                "a_avg_reward": round(float(sub["a_reward"].mean()), 4),
                "b_avg_reward": round(float(sub["b_reward"].mean()), 4),
                "a_avg_total_ships": round(float(sub["a_total_ships"].mean()), 2),
                "b_avg_total_ships": round(float(sub["b_total_ships"].mean()), 2),
                "a_avg_planets": round(float(sub["a_planets"].mean()), 2),
                "b_avg_planets": round(float(sub["b_planets"].mean()), 2),
                "a_avg_production": round(float(sub["a_production"].mean()), 2),
                "b_avg_production": round(float(sub["b_production"].mean()), 2),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["a_win_rate", "b_win_rate"], ascending=[False, True])
        .reset_index(drop=True)
    )


def plot_overview(summary: pd.DataFrame, all_turns: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return

    labels = [f"{r.agent_a}\nvs\n{r.agent_b}" for r in summary.itertuples()]
    x = list(range(len(summary)))
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    axes[0, 0].bar([i - width / 2 for i in x], summary["a_win_rate"], width, label="agent_a")
    axes[0, 0].bar([i + width / 2 for i in x], summary["b_win_rate"], width, label="agent_b")
    axes[0, 0].set_title("Win rate")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].bar([i - width / 2 for i in x], summary["a_avg_reward"], width, label="agent_a")
    axes[0, 1].bar([i + width / 2 for i in x], summary["b_avg_reward"], width, label="agent_b")
    axes[0, 1].set_title("Average reward")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=25, ha="right")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].bar([i - width / 2 for i in x], summary["a_avg_total_ships"], width, label="agent_a")
    axes[1, 0].bar([i + width / 2 for i in x], summary["b_avg_total_ships"], width, label="agent_b")
    axes[1, 0].set_title("Average final ships")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=25, ha="right")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend()

    curves = (
        all_turns.groupby(["matchup", "agent", "turn"], as_index=False)["total_ships"]
        .mean()
        .sort_values(["matchup", "agent", "turn"])
    )

    for (matchup, agent), sub in curves.groupby(["matchup", "agent"]):
        axes[1, 1].plot(sub["turn"], sub["total_ships"], label=f"{matchup} | {agent}")

    axes[1, 1].set_title("Average ships over time")
    axes[1, 1].set_xlabel("turn")
    axes[1, 1].set_ylabel("ships")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=7)

    fig.suptitle(f"Orbit Wars evaluation | seeds={len(SEEDS)}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_replay(out_dir: Path) -> None:
    if not SAVE_ONE_REPLAY_HTML:
        return

    agents = MATCHUPS[REPLAY_MATCHUP_INDEX]

    try:
        env = make(ENV_NAME, configuration={"seed": REPLAY_SEED}, debug=DEBUG)
        env.run(agents)

        html = env.render(mode="html", width=HTML_WIDTH, height=HTML_HEIGHT)
        name = f"replay_matchup_{REPLAY_MATCHUP_INDEX:02d}_seed_{REPLAY_SEED}.html"
        (out_dir / name).write_text(html, encoding="utf-8")
        print(f"Replay saved: {out_dir / name}")

    except Exception as exc:
        print(f"Replay render skipped: {type(exc).__name__}: {exc}")


def run_jobs_sequential(jobs: list[tuple[int, list[str], int]]) -> list[dict]:
    results = []

    current_matchup = None
    for job in jobs:
        matchup_index, agents, seed = job
        matchup = matchup_label(agents)

        if matchup != current_matchup:
            current_matchup = matchup
            print(f"\nRunning: {matchup}", end=" ")

        result = run_one_job(job)
        results.append(result)
        print("." if result["ok"] else "E", end="", flush=True)

    print()
    return results


def run_jobs_parallel(jobs: list[tuple[int, list[str], int]]) -> list[dict]:
    results = []

    print(f"\nRunning {len(jobs)} games with {MAX_WORKERS} workers")
    print("Progress: ", end="", flush=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(run_one_job, job) for job in jobs]

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "ok": False,
                    "matchup_index": None,
                    "seed": None,
                    "agents": None,
                    "matchup": "unknown",
                    "seed_row": None,
                    "final": None,
                    "turns": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

            results.append(result)
            print("." if result["ok"] else "E", end="", flush=True)

    print()
    return results


def collect_results(results: list[dict]):
    seed_rows = []
    player_raw = []
    turns = []
    errors = []

    for result in results:
        if result["ok"]:
            seed_rows.append(result["seed_row"])
            player_raw.append(result["final"])
            turns.append(result["turns"])
        else:
            agents = result.get("agents") or ["unknown", "unknown"]

            errors.append(
                {
                    "matchup_index": result.get("matchup_index"),
                    "matchup": result.get("matchup"),
                    "seed": result.get("seed"),
                    "agent_a": agent_label(agents[0]) if isinstance(agents, list) else "unknown",
                    "agent_b": agent_label(agents[1]) if isinstance(agents, list) else "unknown",
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                    "traceback": result.get("traceback"),
                }
            )

    seed_results = pd.DataFrame(seed_rows)
    player_raw_df = pd.concat(player_raw, ignore_index=True) if player_raw else pd.DataFrame()
    turns_df = pd.concat(turns, ignore_index=True) if turns else pd.DataFrame()
    errors_df = pd.DataFrame(errors)

    return seed_results, player_raw_df, turns_df, errors_df


def main() -> None:
    out_dir = make_run_dir(OUT_ROOT)
    save_config(out_dir)
    print_config(out_dir)

    jobs = build_jobs()

    if PARALLEL and len(jobs) > 1:
        results = run_jobs_parallel(jobs)
    else:
        results = run_jobs_sequential(jobs)

    seed_results, player_raw_df, turns_df, errors_df = collect_results(results)
    summary = summarize_matchups(seed_results, errors_df)

    write_csv(summary, out_dir / "matchup_summary.csv")
    write_csv(seed_results, out_dir / "seed_results.csv")

    if not errors_df.empty:
        write_csv(errors_df, out_dir / "eval_errors.csv")

    if SAVE_PLAYER_RAW and not player_raw_df.empty:
        write_csv(player_raw_df, out_dir / "player_raw_debug.csv")

    if SAVE_TURN_CURVES and not turns_df.empty:
        write_csv(turns_df, out_dir / "turn_curves_debug.csv")

    if not turns_df.empty:
        plot_overview(summary, turns_df, out_dir / "eval_overview.png")

    save_replay(out_dir)

    print("\nMATCHUP SUMMARY")
    if summary.empty:
        print("No successful games.")
    else:
        print(summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {out_dir / 'matchup_summary.csv'}")
    print(f"  {out_dir / 'seed_results.csv'}")
    print(f"  {out_dir / 'eval_overview.png'}")
    print(f"  {out_dir / 'config.json'}")
    if not errors_df.empty:
        print(f"  {out_dir / 'eval_errors.csv'}")
    if SAVE_PLAYER_RAW:
        print(f"  {out_dir / 'player_raw_debug.csv'}")
    if SAVE_TURN_CURVES:
        print(f"  {out_dir / 'turn_curves_debug.csv'}")

    print(f"\nLatest run path saved in: {OUT_ROOT / 'latest_run.txt'}")


if __name__ == "__main__":
    main()