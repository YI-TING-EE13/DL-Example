# Classroom contribution workflow

Practice Git by contributing two short explanations. The core lesson takes about 60 minutes; an optional ML extension can follow. Reserve about 21 minutes for the first contribution after preclass setup. Allow 75 minutes if installation or sign-in must happen in class. Timing is a teaching plan, not a measured classroom result.

## Required tools and optional ML setup

| Required for the Git task | Optional for ML exercises |
|---|---|
| Git, GitHub account, editor, terminal | uv and Python 3.12 |
| Fork, local clone, Git name/email | One PyTorch CPU or CUDA extra |
| Authentication for pushing to your fork | Synthetic environment check and MNIST smoke run |

Windows and PowerShell are the primary classroom environment. Use Git for Windows with Git Credential Manager (GCM). Open a new PowerShell window after installation:

```powershell
git --version
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Replace the name and email. These settings label commits; they do not sign you in. GitHub offers a no-reply email for public commits.

## Fork and clone

1. Sign in to GitHub and open [DL-Example](https://github.com/YI-TING-EE13/DL-Example).
2. Click **Fork**, set **Owner** to your own account, and keep the name **DL-Example**. Copying only **main** is sufficient. Click **Create fork**.
3. Check that your fork's page shows your username as owner. Choose **Code > HTTPS** and copy its URL.
4. In PowerShell, go to your projects folder and clone your fork. Replace `YOUR-USERNAME` below:

```powershell
git clone https://github.com/YOUR-USERNAME/DL-Example.git
cd DL-Example
git remote add upstream https://github.com/YI-TING-EE13/DL-Example.git
git remote -v
```

Know the three locations before using their connection names:

- **Original repository:** the instructor's project on GitHub.
- **Your fork:** your own copy on GitHub.
- **Local clone:** the copy on your computer that you edit.

`origin` points to your fork because you cloned it. `upstream` is the connection name you assigned to the original project. A fork is a separate GitHub repository; a branch is a name inside a repository. Students need write access to their own forks.

## Check HTTPS sign-in

```powershell
git push --dry-run origin main
```

This previews a push without changing your fork. If GCM opens a browser, sign in with the account that owns the fork, approve the requested Git access, and complete two-factor authentication if prompted. Return to PowerShell afterward. Git name/email are separate from authentication. GitHub account passwords are not used as HTTPS Git passwords.

**This is a preliminary check. Your first actual push will confirm that you can publish changes to your fork.** A dry run does not guarantee that every later push or server-side check will succeed. If the first real push fails authentication, diagnose the message with the instructor; it does not by itself show incorrect staging or committing.

For troubleshooting, use GitHub's [credential instructions](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git). The [official fork guide](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo) illustrates the web steps. Other operating systems may have different installation and credential steps.

## Understand staging with a tracked file

The instructor prepares a separate disposable repository with a committed `notes.txt` containing `value=A`. Edit/save B, inspect ordinary diff, add the file, and inspect staged diff. Then save C without adding again. Predict the next ordinary commit before running it:

- Ordinary diff compares index B with working C.
- Staged diff compares HEAD A with index B.
- Commit records B; the working file remains C.

Adding only `notes.txt` does not clear a README change staged earlier. The entire staged diff determines the proposed change. An empty ordinary `git diff` can coexist with staged changes and untracked files; use status to check both.

After committing B, add C and run `git restore --staged notes.txt`. The index returns to B while working C remains. For a newly staged file, unstage leaves it untracked. Plain `git restore <path>` can discard unstaged edits; use it only as an optional recovery demonstration on disposable content.

### Repeat the experiment yourself

Use the Git identity from preclass setup. In PowerShell, go to your projects folder, **outside DL-Example and any other repository**. Choose a new folder name for each attempt:

```powershell
mkdir git-state-lab-01
cd git-state-lab-01
git init -b main
Set-Content -Encoding ascii notes.txt "value=A"
git add notes.txt
git commit -m "Prepare baseline A"
git status
```

Status should report a clean working tree. For another attempt, change the folder name in both commands to `git-state-lab-02`. Do not reinitialize the previous exercise folder. You now have the tracked A baseline. Use the staging example above: save B, add it, then save C without adding again. Predict and compare before committing. `git show HEAD:notes.txt` reads the recorded version; `Get-Content notes.txt` reads the working file. To repeat from A, use a new folder rather than resetting existing work. A clone already contains Git data; do not initialize a nested repository inside it.

## Hands-on Part 1: local Git

Return to your DL-Example clone in PowerShell; the staging lab is a separate repository. Begin with a clean working tree. Update local main, update your fork's main, then create the task branch:

```powershell
git status
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
git switch -c docs/classroom-note
```

Fetch downloads commits and updates your local record of the original main branch (`upstream/main`); it does not change your working files. The fast-forward merge updates local main when history permits. If a command fails, read its message and inspect status before continuing. Ask the instructor about divergent history; a force push or destructive reset is not a general recovery step.

The contribution then follows this sequence:

**Choose task > create branch > edit/save > inspect/add/verify > commit locally > push to fork > open PR > review/merge.**

Use the update/start sequence once per new task. Do not repeatedly create an existing branch.

### Create a new documentation file

Create `docs/classroom-notes/YOUR-USERNAME.md` in your editor, making the directory if needed. Write one sentence explaining editor save versus commit, and one explaining commit versus push. Use your own GitHub username and explanation; include no personal contact information or training artifacts.

This file starts **untracked**. It differs from the tracked `notes.txt` staging demonstration above. Inspect and select it:

```powershell
git status
git diff
git add docs/classroom-notes/YOUR-USERNAME.md
git diff --staged
```

Status lists a new untracked file, but ordinary diff does not show its content. Read it in your editor first. After add, read the staged diff and verify the intended file and text.

**Pause before commit: What will your next commit contain?** Point to the selected file and lines. Remember that a staged diff shows changes from HEAD, not the entire snapshot; unchanged tracked contents remain included.

```powershell
git commit -m "docs: explain local and shared version records"
git status
```

This completes the local part. Your commit has not yet been published.

## Hands-on Part 2: GitHub collaboration

In the same clone and on the same task branch, use PowerShell to publish the branch:

```powershell
git push -u origin docs/classroom-note
```

Keep any authentication error message for diagnosis. After a successful push, continue on GitHub.

**Pause after push: Where does your committed change exist now?** Name every location containing it. Explain whether the original main branch has changed and what must happen before it does.

### Prepare a pull request

On the original repository, choose **Pull requests > New pull request**, then **Compare across forks** when needed:

| Question | Selection |
|---|---|
| Where should the change go? | **Base:** `YI-TING-EE13/DL-Example`, branch `main` |
| Where does the change come from? | **Head:** your fork, branch `docs/classroom-note` |

### Review the diff, then create the PR

Review the displayed diff and confirm that it contains only the intended file and changes. Select **Create pull request**, add a short title and description of what changed and how you checked it, then submit the PR. A reviewer checks the sentences for conceptual accuracy; a synthetic Python check is not required for this task.

### Record and revise your result

Open the submitted PR and record its URL, branch name, and commit SHA. Show its diff and your answers to both pause questions. Respond to feedback by committing and pushing to the same task branch; the PR updates. Creating a PR proposes integration; it does not merge it. Maintainer review, approval, and merge are separate events that can happen after class. Fork CI may also require maintainer approval. An open PR is sufficient for classroom submission.

## Optional ML extension

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and check `uv --version`. The project requires Python 3.12. Follow the [README installation steps](../README.md#install) to select exactly one of `cpu` or `cuda`; uv uses the project's Python pin and lockfile. A compatible NVIDIA setup uses CUDA automatically, while CPU execution remains available.

After installation:

```powershell
uv run --no-sync python --version
uv run --no-sync python -m dl_example check
uv run --no-sync python run_mnist.py --smoke
```

Confirm Python 3.12. The synthetic check uses no dataset. The smoke run downloads the complete source MNIST dataset on first use despite training on subsets; prepare it before class.

Inspect the run's actual settings, code revision, split/input manifest, environment, and summary. Two students can use the same commit with different ignored data or command-line settings and obtain different results. Full MNIST training and CIFAR-10 remain separate from the Git task. See [verification](verification.md) for the measured validation scope.
