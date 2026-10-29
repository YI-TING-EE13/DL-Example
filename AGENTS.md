# Agent guidance

Read README.md and docs/architecture.md before editing. Use the project uv environment on Windows or the declared CI environment; select exactly one of the cpu or cuda extras. Preserve the two root training entry scripts and both model architectures unless a task explicitly changes them.

Validate configuration before dataset downloads. Place model parameters on the target device before creating the optimizer. Keep the official test split out of model selection. Preserve the distinction between smoke execution, full training, numerical repeatability, and historical result reproduction.

Run pytest, Ruff lint, and Ruff formatting checks for code changes. Document runtime checks with commands, environment, and artifacts. Keep code comments and maintained documents in English. Do not claim resume support without a restore test. Keep runtime data and credentials out of commits. Do not rewrite existing Git history or force-push.
