import sys
import json
import csv
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode
import importlib.util

spec = importlib.util.spec_from_file_location("user_agent", ROOT / "agent" / "my_agent.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MyAgentCls = module.MyAgent

try:
    from agent.baselines import HUMAN
except ImportError:
    HUMAN = {}

def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()
    game_ids = [e.game_id.split("-")[0] for e in all_envs]
    
    results = []
    
    print(f"Running baseline audit on {len(game_ids)} games...")
    
    for i, game_id in enumerate(game_ids, 1):
        print(f"[{i}/{len(game_ids)}] Playing {game_id}...")
        env = arc.make(game_id, render_mode=None)
        if env is None:
            continue
            
        agent = MyAgentCls(
            card_id="local-dev",
            game_id=game_id,
            agent_name=f"MyAgent.local.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["local-dev"],
        )
        
        t0 = time.time()
        try:
            agent.main()
            error = None
        except Exception as e:
            error = str(e)
        t1 = time.time()
        runtime = t1 - t0
        
        final = agent.frames[-1]
        levels_cleared = final.levels_completed
        # The game has some number of levels in HUMAN. If unknown, we don't know total levels.
        total_levels = len(HUMAN.get(game_id, []))
        
        # Calculate level score based on HUMAN
        human_actions = HUMAN.get(game_id, [])
        level_costs = getattr(agent, "level_costs", {})
        game_score = 0.0
        
        if total_levels > 0:
            sum_weighted = 0.0
            sum_denom = sum(range(1, total_levels + 1))
            for lv in range(1, levels_cleared + 1):
                if lv in level_costs and lv <= len(human_actions):
                    h_act = human_actions[lv - 1]
                    a_act = level_costs[lv]
                    eff = h_act / a_act if a_act else 0
                    score = min(1.0, eff) ** 2
                    sum_weighted += lv * score
            game_score = sum_weighted / sum_denom if sum_denom else 0.0

        res = {
            "game_id": game_id,
            "levels_attempted": total_levels,
            "levels_cleared": levels_cleared,
            "actions": agent.action_counter,
            "human_actions_total": sum(human_actions[:levels_cleared]) if human_actions else None,
            "game_score": game_score,
            "runtime": runtime,
            "runtime_per_action": runtime / max(1, agent.action_counter),
            "LLM_calls": getattr(agent, "model_calls", 0),
            "tokens": None, # Not measured by agent currently
            "errors": error,
            "termination_reason": str(final.state)
        }
        results.append(res)
        print(f"  -> {levels_cleared}/{total_levels} levels, {agent.action_counter} actions, score {game_score:.4f}, state {final.state}")
        
    out_dir = Path("experiments/000_baseline_audit")
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    if results:
        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    print("Baseline audit complete.")

if __name__ == "__main__":
    main()

