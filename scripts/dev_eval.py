"""
Simple Orbit Wars evaluation.

No argparse. Edit the constants below and run from project root:
    python scripts/dev_eval.py

Default output is intentionally small:
    outputs/dev_eval/eval_summary.csv
    outputs/dev_eval/eval_overview.png
    outputs/dev_eval/replay.html

Optional raw output:
    set SAVE_RAW_RESULTS = True to also save eval_raw.csv.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Keep Kaggle/OpenSpiel optional-info spam out of the terminal.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("kaggle_environments.envs.open_spiel_env.open_spiel_env").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from kaggle_environments import make  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from owtools.metrics import final_rows_from_env, rows_from_env, save_csv  # noqa: E402


# =====================
# EDIT ONLY THIS BLOCK
# =====================
ENV_NAME = "orbit_wars"
DEBUG = True
SEEDS = list(range(20))
OUT_DIR = Path("outputs/dev_eval")

# main.py is your submission entrypoint.
# It currently imports agents.roi_expansion.agent.
MATCHUPS = [
    ["main.py", "random"],
    ["main.py", "agents/nearest_sniper.py"],
    ["main.py", "agents/passive.py"],
]

SAVE_RAW_RESULTS = False      # False = only one summary CSV
SAVE_ONE_REPLAY_HTML = True   # only one replay, not one per seed
REPLAY_MATCHUP_INDEX = 0
REPLAY_SEED = SEEDS[0]
HTML_WIDTH = 1200
HTML_HEIGHT = 850
# =====================


def agent_label(agent_path: str) -> str:
    if agent_path in {"random", "reaction"}:
        return agent_path
    if agent_path == "main.py":
        return "main_submission"
    return Path(agent_path).with_suffix("").as_posix().replace("/", ".")


def matchup_label(agents: list[str]) -> str:
    return " vs ".join(agent_label(a) for a in agents)


def print_config() -> None:
    print("=" * 90)
    print(f"ENV      : {ENV_NAME}")
    print(f"SEEDS    : {SEEDS}")
    print(f"OUT_DIR  : {OUT_DIR.resolve()}")
    print("MATCHUPS :")
    for i, agents in enumerate(MATCHUPS):
        print(f"  {i}: {matchup_label(agents)} -> {agents}")
    print("=" * 90)


def run_match(agents: list[str], seed: int):
    env = make(ENV_NAME, configuration={"seed": seed}, debug=DEBUG)
    env.run(agents)

    labels = dict(enumerate(agent_label(a) for a in agents))
    matchup = matchup_label(agents)

    final = final_rows_from_env(env, seed)
    final["matchup"] = matchup
    final["agent"] = final["player"].map(labels)

    turns = rows_from_env(env)
    turns["seed"] = seed
    turns["matchup"] = matchup
    turns["agent"] = turns["player"].map(labels)

    return env, final, turns


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["matchup", "player", "agent"], as_index=False)
        .agg(
            games=("seed", "count"),
            wins=("rank", lambda s: int((s == 1).sum())),
            win_rate=("rank", lambda s: round(float((s == 1).mean()), 4)),
            avg_rank=("rank", lambda s: round(float(s.mean()), 3)),
            avg_reward=("reward", lambda s: round(float(s.mean()), 3)),
            avg_total_ships=("total_ships", lambda s: round(float(s.mean()), 2)),
            avg_planets=("planet_count", lambda s: round(float(s.mean()), 2)),
            avg_production=("production", lambda s: round(float(s.mean()), 2)),
        )
        .sort_values(["matchup", "win_rate", "avg_reward"], ascending=[True, False, False])
    )
    return summary


def plot_overview(summary: pd.DataFrame, all_turns: pd.DataFrame, out_path: Path) -> None:
    """One image with the whole evaluation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compact label per row.
    labels = [f"{r.matchup}\nP{r.player}: {r.agent}" for r in summary.itertuples()]
    x = list(range(len(summary)))

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    axes[0, 0].bar(x, summary["win_rate"])
    axes[0, 0].set_title("Win rate")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(x, summary["avg_reward"])
    axes[0, 1].set_title("Average reward")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=35, ha="right")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(x, summary["avg_total_ships"])
    axes[1, 0].set_title("Average final ships")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=35, ha="right")
    axes[1, 0].grid(axis="y", alpha=0.25)

    # One combined time curve: average total ships over all seeds per matchup/agent.
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
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"Orbit Wars evaluation | seeds={len(SEEDS)}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def maybe_save_replay(env, matchup_index: int, seed: int) -> None:
    if not SAVE_ONE_REPLAY_HTML:
        return
    if matchup_index != REPLAY_MATCHUP_INDEX or seed != REPLAY_SEED:
        return

    try:
        html = env.render(mode="html", width=HTML_WIDTH, height=HTML_HEIGHT)
        (OUT_DIR / "replay.html").write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"Replay render skipped: {exc}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print_config()

    finals = []
    turns = []

    for matchup_index, agents in enumerate(MATCHUPS):
        print(f"\nRunning: {matchup_label(agents)}")
        for seed in SEEDS:
            env, final, per_turn = run_match(agents, seed)
            finals.append(final)
            turns.append(per_turn)
            maybe_save_replay(env, matchup_index, seed)

            row = final[["seed", "player", "agent", "rank", "reward", "total_ships", "planet_count", "production"]]
            print(row.to_string(index=False))

    raw = pd.concat(finals, ignore_index=True)
    all_turns = pd.concat(turns, ignore_index=True)
    summary = summarize(raw)

    save_csv(summary, OUT_DIR / "eval_summary.csv")
    if SAVE_RAW_RESULTS:
        save_csv(raw, OUT_DIR / "eval_raw.csv")

    plot_overview(summary, all_turns, OUT_DIR / "eval_overview.png")

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"  {OUT_DIR / 'eval_summary.csv'}")
    print(f"  {OUT_DIR / 'eval_overview.png'}")
    if SAVE_ONE_REPLAY_HTML:
        print(f"  {OUT_DIR / 'replay.html'}")
    if SAVE_RAW_RESULTS:
        print(f"  {OUT_DIR / 'eval_raw.csv'}")


if __name__ == "__main__":
    main()
