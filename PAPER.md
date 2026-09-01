# Programmatic/Selective-Model World Modelling Under a Hard Wall-Clock Budget

## Abstract
[PLACEHOLDER: Abstract text summarizing the thesis, approach, and primary results.]

## The 430ms Problem
The ARC-AGI-3 competition imposes strict wall-clock budgets on agent execution. Given the constraints, agents have roughly 430ms per action. At this scale, treating every action as an LLM generation step inevitably fails the budget, as network latency and inference time for a capable model consistently exceed 430ms. This bottleneck necessitates a shift away from pure per-action LLM calls toward a hybrid approach.

## Architecture
To address the 430ms problem, we propose an architecture that leverages programmatic heuristics for the majority of execution steps, reserving model consultation for rare, high-leverage decisions.
[PLACEHOLDER: Describe the heuristics used and the criteria for when the model is consulted.]

## Results
[PLACEHOLDER: Include performance metrics, such as median score, average actions per level, and success rates, derived from `eval_runs.py` and `level_report.md`.]

## Related Work
Our hybrid approach builds upon and contrasts with recent work in agent architectures and world modeling:
- **OPINE-World** (arXiv:2607.01531): [PLACEHOLDER: Brief discussion on how OPINE-World relates to our methodology.]
- **Tiny Recursion Model** (arXiv:2511.02886): [PLACEHOLDER: Connection to recursive model interactions or efficiency.]
- **ARC-AGI-3 Technical Report** (arXiv:2603.24621): Contextualizes the competition constraints and baseline expectations that motivate our architectural choices.

