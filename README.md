# Orbit Wars - simple local evaluation project

## What matters

- `main.py` is the Kaggle submission entrypoint.
- `main.py` currently uses `agents/roi_expansion.py`.
- `scripts/dev_eval.py` runs local tests without argparse.
- Output is intentionally small: one CSV, one PNG, one replay HTML.

## Install

```bash
pip install -r requirements.txt
```

## Run simple evaluation

```bash
python scripts/dev_eval.py
```

Default outputs:

```text
outputs/dev_eval/eval_summary.csv
outputs/dev_eval/eval_overview.png
outputs/dev_eval/replay.html
```

`eval_summary.csv` is the only default CSV. It contains one row per matchup/player/agent with:

- games
- wins
- win_rate
- avg_rank
- avg_reward
- avg_total_ships
- avg_planets
- avg_production

`eval_overview.png` contains the whole comparison in one image.

## Choose tests

Edit only this block in `scripts/dev_eval.py`:

```python
SEEDS = list(range(20))

MATCHUPS = [
    ["main.py", "random"],
    ["main.py", "agents/nearest_sniper.py"],
    ["main.py", "agents/passive.py"],
]
```

More seeds = more stable result.

## Change submitted agent

`main.py` is intentionally tiny:

```python
from agents.roi_expansion import agent
```

To submit another strategy, change that import, for example:

```python
from agents.nearest_sniper import agent
```

For the current intended setup, leave it as `roi_expansion`.

## Create submission

```bash
python scripts/make_submission.py
```

Then:

```bash
kaggle competitions submit orbit-wars -f submissions/submission.tar.gz -m "roi expansion"
```
