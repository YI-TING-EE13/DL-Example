# Classroom contribution workflow

Use this project to practice Git while making one small, reviewable change. Prepare Git, GitHub authentication, uv, the selected compute extra, and the synthetic check before class. Model training is an optional extension to the Git lesson.

## Fork and clone

1. Open [DL-Example](https://github.com/YI-TING-EE13/DL-Example) and choose **Fork**. Keep the default repository name.
2. Clone your fork. Replace `YOUR-USERNAME` with your actual GitHub username:

```powershell
git clone https://github.com/YOUR-USERNAME/DL-Example.git
cd DL-Example
git remote add upstream https://github.com/YI-TING-EE13/DL-Example.git
git remote -v
```

`origin` is your fork because that is what you cloned. `upstream` is a conventional name you assign to the instructor's original repository. A fork is a separate GitHub repository; a branch is a name within a repository. You do not need write access to the instructor's repository to push to your own fork.

## Start one task

Start with a clean working tree. Update local main from the original repository, then create a task branch:

```powershell
git status
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
git switch -c docs/classroom-note
```

Here `fetch` updates your local knowledge of upstream history. The explicit fast-forward merge updates local main. If Git reports divergent history, inspect it with the instructor; do not force-push.

## A small documentation task

The instructor can assign: **Create `docs/classroom-notes/YOUR-USERNAME.md` with one sentence explaining editor save versus commit, and one sentence explaining commit versus push.** The file's contents must be your own explanation. Do not add personal contact information or training artifacts.

Inspect, select, verify, and commit the task file:

```powershell
git status
git diff
git add docs/classroom-notes/YOUR-USERNAME.md
git diff --staged
uv run --no-sync python -m dl_example check
git commit -m "docs: explain local and shared version records"
git push -u origin docs/classroom-note
```

A newly created untracked file is listed by status, but ordinary `git diff` does not show its contents. Inspect it in the editor, then inspect the staged diff after add. The synthetic check verifies the environment; the reviewer must still read the explanation for conceptual accuracy.

## Open and revise a pull request

On GitHub choose **Compare across forks** when needed:

- Base repository: `YI-TING-EE13/DL-Example`; base branch: `main`.
- Head repository: your fork; compare branch: `docs/classroom-note`.

Explain what changed and how you checked it. Respond to feedback by committing and pushing to the same task branch; the open PR updates. An open PR is the minimum classroom outcome. The instructor controls merge decisions. Classroom PRs are not guaranteed to be merged.

## Understand staging before practice

In a disposable practice branch, create a text file, commit a baseline value, change it to B, add it, then change it to C without adding again. Predict the next commit's content, compare `git diff` and `git diff --staged`, and check the committed content. That commit records B; the working file remains C.

Use `git restore --staged <path>` to reset the staging selection while preserving edits. `git restore <path>` replaces a tracked working file from the index and can discard an unstaged edit. Inspect first and use disposable examples for destructive practice.

## Optional ML extension

Run the MNIST smoke configuration. Find the run's configuration, code revision, split manifest, and summary. Explain why changing an ignored local dataset or running uncommitted code cannot be described completely by a commit SHA. Full MNIST training and CIFAR-10 are separate from the 15-minute Git task.
