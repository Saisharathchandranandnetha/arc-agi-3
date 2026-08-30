# Issues Found During Baseline Audit

1. **Missing Files Mentioned in Roadmap**:
   - `rhae.py`
   - `budget.py`
   Neither of these files exists in the current repository, despite `ROADMAP.md` indicating they were completed in Phase 0.

2. **Replay Verifier / World Model**:
   - No code actually implements `verify(program, buffer)` or world modeling.
   - The LLM integration (`_consult_model`) just asks the model to suggest an action string rather than writing Python transition code.

3. **Incomplete Trace Logging**:
   - The agent keeps `_history` limited to the last 10 actions and does not store a full, exact replay trace matching the state transitions perfectly for later verification.
