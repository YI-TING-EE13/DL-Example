# Contributing

Discuss a focused task, create a branch, and submit a pull request to `main`. For a classroom note, use the [minimal contribution entry](docs/classroom-notes/README.md) and the instructor-provided Git course workflow. Classroom submissions are assessed separately and may be closed without merging. Keep source changes, experiment-method changes, and documentation updates distinguishable.

After installing one compute extra, run from the repository root:

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

The pull request should explain the problem, resulting behavior, checks performed, and remaining limitations. Include configuration or checkpoint compatibility changes when relevant. Write documentation and comments in clear professional English; use instructional language for learner-facing exercises.

Do not commit datasets, weights, local run directories, environments, credentials, or personal machine paths. Commit `uv.lock` when dependencies change. A changed training protocol requires a new result record; do not relabel old results.
