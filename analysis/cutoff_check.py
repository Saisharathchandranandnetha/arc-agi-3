#!/usr/bin/env python3
import sys
import re
from pathlib import Path

# Add project root to sys.path to import agent.baselines
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from agent.baselines import HUMAN
except ImportError:
    print("Error: Could not import agent.baselines.HUMAN")
    sys.exit(1)

CLEAR_RE = re.compile(
    r"\[CLEAR\]\s+(\w+)\s+L(\d+)\s+in\s+(\d+)\s+actions"
)

def main():
    if len(sys.argv) > 1:
        f = open(sys.argv[1], "r")
    else:
        f = sys.stdin

    print("Checking for levels exceeding 5x human baseline...")
    found = False
    
    for line in f:
        m = CLEAR_RE.search(line)
        if m:
            game = m.group(1)
            level = int(m.group(2))
            actions = int(m.group(3))
            
            if game in HUMAN:
                if level < len(HUMAN[game]):
                    human_actions = HUMAN[game][level]
                    cutoff = human_actions * 5
                    if actions > cutoff:
                        print(f"FLAG: {game} L{level} took {actions} actions (Human: {human_actions}, Cutoff: {cutoff}) -> WASTED (Score 0)")
                        found = True
    
    if not found:
        print("All clears within 5x human baseline!")

if __name__ == "__main__":
    main()

