import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode

# Load agent
import importlib.util
spec = importlib.util.spec_from_file_location("user_agent", ROOT / "agent" / "my_agent.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MyAgentCls = module.MyAgent

def main():
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    game_id = "ls20"  # A known game
    env = arc.make(game_id, render_mode=None)
    
    agent = MyAgentCls(
        card_id="local-dev",
        game_id=game_id,
        agent_name=f"MyAgent.local.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["local-dev"],
    )
    
    # We will override take_action to record the trace
    trace = {"game_id": game_id, "steps": []}
    original_take_action = agent.take_action
    
    def hooked_take_action(action):
        obs_before = [arr.tolist() for arr in env.observation_space.frame] if env.observation_space and hasattr(env.observation_space, 'frame') else None
        
        # Call the original which interacts with arc_env
        frame = original_take_action(action)
        
        obs_after = frame.frame if frame else None
        
        step_data = {
            "step": agent.action_counter,
            "observation": obs_before,
            "action": action.name,
            "action_data": getattr(action, "_data", None),
            "next_observation": obs_after
        }
        trace["steps"].append(step_data)
        return frame
        
    agent.take_action = hooked_take_action
    agent.MAX_ACTIONS = 10  # Just enough for a trace
    agent.main()
    
    with open("experiments/000_baseline_audit/one_game_trace.json", "w") as f:
        json.dump(trace, f, indent=2)

if __name__ == "__main__":
    main()
