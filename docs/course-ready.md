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
