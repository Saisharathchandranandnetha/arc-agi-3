#!/usr/bin/env python3
"""
eval_runs.py - Run the agent N times, average the score, report the spread.

The single most important tool in this repo. On ARC-AGI-3 the run-to-run
variance is enormous (we've seen 0.024 to 0.084 on IDENTICAL code), because
the agent explores stochastically. A single make play-local tells you almost
nothing. This runs it N times and gives you mean +/- std, so a change you make
can be judged against the NOISE, not against one lucky or unlucky run.

Usage:
    python eval_runs.py --n 5
    python eval_runs.py --n 5 --max-steps 600 --tag "guard-on"

It shells out to the same play_local.py you already use, scrapes the
"Aggregate scorecard score" line, and does the statistics.

License: CC0-1.0
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


SCORE_RE = re.compile(r"Aggregate scorecard score:\s*([0-9.]+)")
LEVEL_RE = re.compile(r"^\s*(\w+)\s+levels=\s*(\d+)\s+actions=\s*(\d+)", re.M)


def venv_python() -> str:
    """Absolute path to the project venv's interpreter, for THIS platform.

    `python -m venv` puts the interpreter in `Scripts\\python.exe` on Windows
    and `bin/python` everywhere else. Hard-coding either one makes this script
    non-portable, which is why it previously failed on a Windows checkout.

    Falls back to the interpreter running this script if it is already inside
    the venv (so `python eval_runs.py` works after `activate`). Exits with an
    actionable message rather than letting subprocess raise a bare
    FileNotFoundError two frames deeper.
    """
    root = Path(__file__).resolve().parent
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",   # Windows
        root / ".venv" / "bin" / "python",           # POSIX
    ]
    if os.name != "nt":
        candidates.reverse()                          # try the native one first

    for c in candidates:
        if c.is_file():
            return str(c)

    # Already running inside the venv? Then sys.executable is the right answer.
    here = Path(sys.executable).resolve()
    if (root / ".venv").resolve() in here.parents:
        return str(here)

    sys.exit(
        "No virtualenv interpreter found. Looked for:\n"
        + "".join(f"  {c}\n" for c in candidates)
        + "Run `make setup` (or `python -m venv .venv` and install "
          "requirements) from the repo root first."
    )


def one_run(max_steps: int, timeout: int) -> tuple[float, dict, list]:
    """Run play_local.py once. Return (score, {game: levels}, clears)."""
    cmd = [venv_python(), "scripts/play_local.py", "--max-steps", str(max_steps)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    m = SCORE_RE.search(out)
    score = float(m.group(1)) if m else float("nan")
    levels = {g: int(lv) for g, lv, _ in LEVEL_RE.findall(out)}
    
    # Parse [CLEAR] lines
    # Format: [CLEAR] {game} L{n} in {k} actions (human {h}, ratio {r}, level_score {s})
    clear_re = re.compile(
        r"\[CLEAR\]\s+(\w+)\s+L(\d+)\s+in\s+(\d+)\s+actions\s+\(human\s+(\d+),\s+ratio\s+([0-9.]+),\s+level_score\s+([0-9.]+)\)"
    )
    clears = []
    for match in clear_re.finditer(out):
        clears.append({
            "game": match.group(1),
            "level": int(match.group(2)),
            "actions": int(match.group(3)),
            "level_score": float(match.group(6))
        })
        
    return score, levels, clears


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of runs")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--timeout", type=int, default=1800, help="per-run seconds")
    ap.add_argument("--tag", type=str, default="", help="label for this batch")
    args = ap.parse_args()

    if not Path("scripts/play_local.py").exists():
        sys.exit("run this from the repo root (scripts/play_local.py not found)")

    print(f"{'='*64}")
    print(f"EVAL BATCH{'  ['+args.tag+']' if args.tag else ''}"
          f"  n={args.n}  max_steps={args.max_steps}")
    print(f"{'='*64}")

    scores, all_levels, clears_per_game = [], [], {}
    all_clears = []
    for i in range(args.n):
        t0 = time.time()
        try:
            score, levels, clears = one_run(args.max_steps, args.timeout)
        except subprocess.TimeoutExpired:
            print(f"  run {i+1}: TIMEOUT after {args.timeout}s - lower --max-steps")
            continue
        dt = time.time() - t0
        scores.append(score)
        all_levels.append(levels)
        all_clears.append(clears)
        solved = {g: lv for g, lv in levels.items() if lv > 0}
        for g, lv in solved.items():
            clears_per_game[g] = clears_per_game.get(g, 0) + 1
        print(f"  run {i+1}:  score={score:.4f}  "
              f"solved={len(solved)} games  "
              f"({', '.join(f'{g}:{lv}' for g,lv in sorted(solved.items()))})"
              f"  [{dt:.0f}s]")

    if not scores:
        sys.exit("no successful runs")

    print(f"{'-'*64}")
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"  MEAN score : {mean:.4f}")
    print(f"  STD  score : {sd:.4f}   ({100*sd/mean:.0f}% of mean)" if mean else "")
    print(f"  MIN / MAX  : {min(scores):.4f} / {max(scores):.4f}")
    print(f"  95% CI     : {mean:.4f} +/- {1.96*sd/len(scores)**0.5:.4f}")
    print(f"{'-'*64}")

    # Reliability: which games clear EVERY run vs only sometimes.
    print("  per-game clear reliability (times solved / runs):")
    for g in sorted(clears_per_game, key=lambda k: -clears_per_game[k]):
        n = clears_per_game[g]
        bar = "#" * n + "." * (len(scores) - n)
        tag = "RELIABLE" if n == len(scores) else "flaky" if n > 1 else "lucky"
        print(f"    {g}  [{bar}] {n}/{len(scores)}  {tag}")
    print()
    print("  A change is only real if its MEAN beats the old mean by more than")
    print("  the 95% CI half-width above. Otherwise you're reading noise.")
    print(f"{'='*64}")

    # Generate analysis/level_report.md
    from collections import defaultdict
    
    stats = defaultdict(lambda: defaultdict(lambda: {'actions': [], 'scores': [], 'clears': 0}))
    for run_clears in all_clears:
        for c in run_clears:
            g = c["game"]
            lv = c["level"]
            stats[g][lv]['actions'].append(c["actions"])
            stats[g][lv]['scores'].append(c["level_score"])
            stats[g][lv]['clears'] += 1
            
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/level_report.md", "w") as f:
        f.write("# Per-Level Clear Report\n\n")
        f.write(f"Across {len(scores)} runs.\n\n")
        f.write("| Game | Level | Clears | Median Actions | Median Score |\n")
        f.write("|------|-------|--------|----------------|--------------|\n")
        
        for g in sorted(stats.keys()):
            for lv in sorted(stats[g].keys()):
                d = stats[g][lv]
                med_act = statistics.median(d['actions'])
                med_score = statistics.median(d['scores'])
                f.write(f"| {g} | {lv} | {d['clears']}/{len(scores)} | {med_act:.1f} | {med_score:.3f} |\n")


if __name__ == "__main__":
    main()