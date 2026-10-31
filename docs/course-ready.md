# Course-ready classroom state

The classroom entry is [the contribution workflow](teaching.md). The task is a small documentation contribution: no Python, uv, PyTorch, dataset, or model execution is required. A fresh fork from current main contains the classroom-note folder, its creation instructions, and the five-field submission template. Runtime and experiment behavior are unchanged by this documentation revision.

## Repository and material preparation

- Start or resume only after checking repository root, branch, and status; preserve existing work.
- Use the tracked-file staging demonstration separately from the student's new untracked note.
- Teach unstage as preserving working-file edits; repeat execution independently afterward.
- Create a personal note in [classroom-notes](classroom-notes/README.md), inspect it, stage only its path, and verify the staged diff.
- After push, verify the fork's task branch and file. For a PR, select original/main as base and the student's fork/task branch as head, then review the actual diff.
- Gather branch, actual contribution SHA, and URL temporarily; answer Content and Location; save all five fields together using [the template](classroom-submission.md).
- A blocked push and a successful push with blocked PR creation have different observed locations. Use truthful pending fields and private collection; distinguish receipt, correctness, and publication.
- Use the preferred 75-minute plan with 24 minutes of practice and five for collection. If needed, borrow one or two minutes from the three-minute buffer for saving; retain unresolved cases for follow-up. These are planning estimates.

The existing [CPU CI workflow](../.github/workflows/ci.yml) runs Windows and Ubuntu checks: locked Python 3.12 CPU environment, Ruff lint/format checks, pytest, and a synthetic environment check without a dataset. Require the candidate's checks to pass before merging its PR; verify the merged main and its checks afterward. This documentation update does not establish full MNIST/CIFAR-10 reproduction.

## Delivery checks requiring the actual class

Repository/material readiness does not establish projector readability, every student's credentials/network access, or learning acceptance. Before class, instructors test their real fork/clone/branch/PR, confirm the collection route, prepare separate disposable demo repositories, and distribute the preclass guide. During class, diagnose the last completed step (edit, stage, commit, push, PR, evidence) before treating an issue as authentication.

A dry-run push is preliminary; the first actual push confirms publication access. Do not substitute a teacher demonstration, local simulation, an open PR, or a quick scan for a student's own completed work and demonstrated understanding. Record real classroom timing and follow-up evidence separately.

## Branch authority

The repository default branch, `main`, is the classroom entry. Fork its latest main and create your own `docs/classroom-note` task branch using the contribution guide. Maintainer branch names are not student starting points.

| Ref | Role |
|---|---|
| `main` | Current delivery and classroom source; use its latest tested revision. |
| `archive/classroom-round4-20261030` (tag) | Preserved [PR #1](https://github.com/YI-TING-EE13/DL-Example/pull/1) head `209c5a8bb7da438ae9537f52f276e97fa070c5a6`. Its tree matches main at `c208cf12d8e3a28e574b04a4ae0e3aa8aa7cff11`, before later documentation updates. |
| `archive/course-ready-source-audit-20261030` (tag) | Preserved [PR #2](https://github.com/YI-TING-EE13/DL-Example/pull/2) head `e7debfa0fa12e7c0bef4079606c0670fe0b6f828`. Its tree matches main at `8f491c83d62735bf8e409e161971c2c4b5eaddde`, before this branch-guide update. |
| Implementation commit `d3eac41e9f5c62639ab4370f6f0a422a07cc772a` | Remains an ancestor of main and identifies the historical executable verification. Its separate development branch is unnecessary. |
| Active maintainer work branches | Temporary, focused PR work; read the associated PR state. They do not replace main. |

The three completed development branches can be removed after this guide's PR passes checks and is merged. The two archive tags preserve the unique pre-squash PR histories; the implementation commit remains reachable through main. A squash-merged PR head need not be an ancestor by commit identity; compare the associated merge and content before treating its commits as unfinished work. No original commit history is rewritten. Old snapshots do not redefine the current classroom instructions. Students do not need these tags or a permanent demonstration branch to complete the fork workflow.

## Source, comment and readiness audit

The [2026-10-30 CPU/source audit](validation/course-ready-audit.json) records the audited executable base, hashes of all Python source/tests, configuration/lock checks, and branch classifications. The complete 38-file starting tree was reviewed, including 16 Python files and their comments/docstrings. Maintained documents were compared with commands, checkpoint policy, data split policy and recorded evidence. No runtime or comment correction was required for this course revision.

The locked Windows Python 3.12 CPU environment passed Ruff lint, Ruff format, all 28 tests and a real synthetic forward/backward/optimizer update without a dataset. Required CI must also pass for the PR and merged main. New readiness documentation is reviewed separately; original training evidence remains tied to its recorded executable revision.

This audit does not rerun full MNIST/CIFAR-10 training, GPU repeatability, optional tracking, student authentication or classroom acceptance. The original [verification record](verification.md) continues to describe its own measured environment and results.
