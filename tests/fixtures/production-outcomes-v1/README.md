# Fresh production outcome fixtures

Four new coding tasks, held aside from optimization selection. Run the shared
runner with --case, at T1/P1/top_k=-1/high, to natural completion. It preserves
reasoning/tool identities and measures all model calls and tools through the
final answer. The earlier events and planner fixtures remain separate and
unchanged. These six coding tasks are only part of the intended10–20task panel;
planning, ideation and image tasks still need to be added. No results yet.

Generated task files execute only inside the runner’s bounded, read-only,
networkless Docker sandbox. Reference implementations below are fixture sanity
controls, never given to the served model. Held-out graders are never tools.
