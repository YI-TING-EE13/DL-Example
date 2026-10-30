# Classroom contribution workflow

Practice Git by contributing two short explanations. Prefer a 75-minute first class after preclass setup: 24 minutes for student contribution, five for individual answers and collection, and three for questions/buffer. The prepared 60-minute alternative reserves 21 minutes for contribution and four for collection. Installation or sign-in during the 75-minute class can extend it to 90 minutes without ML activities. Timing is a teaching plan, not a measured classroom result.

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

Use a disposable repository outside DL-Example. During class, observe the instructor in that separate practice repository and make your own predictions. Repeat the execution independently afterward with the self-study instructions below. The branch and GitHub concept pages in the handout are demonstrations; begin your own contribution only at Hands-on Part 1.

Start in a new practice repository with `value=A` already committed in `notes.txt`. The table entries are file contents; HEAD means content in the current commit. Save B, run `git add notes.txt`, then save C without adding again. Before committing, fill the final row independently:

| Action | Working file | Index | HEAD |
|---|---|---|---|
| Initial commit | A | A | A |
| Save B | B | A | A |
| Add notes.txt | B | B | A |
| Save C | C | B | A |
| Your prediction after commit | ___ | ___ | ___ |

Run `git status`. This excerpt omits branch information and hints; tabs are expanded:

```text
Changes to be committed:
    modified:   notes.txt

Changes not staged for commit:
    modified:   notes.txt
```

The first group contains selected changes. The second contains working-file changes not selected for commit; here, these are the edits from B to C. A tracked edit can also appear in this group before add or after unstage. One path can appear in both groups. Inspect both comparisons and point to evidence for your prediction:

| Command | Before -> after |
|---|---|
| `git diff --staged -- notes.txt` | HEAD -> index: A -> B |
| `git diff -- notes.txt` | Index -> working tree: B -> C |

In a patch, `-` marks a removed line and `+` an added line in that comparison. The staged patch includes `-value=A` and `+value=B`; `git diff` includes `-value=B` and `+value=C`.

After the class prediction check, execute the original sequence **without another add**:

```powershell
git commit -m "Record the staged value"
Get-Content notes.txt
git show :notes.txt
git show HEAD:notes.txt
```

Compare the working file, index, and commit with your prediction. Explain any difference.

Now consider one change to that original sequence: what if you had added `notes.txt` again after saving C? Predict the first commit's content; a second execution is not required.

After this check, add the file again and use `git restore --staged notes.txt`. Working edits remain; inspect status. For a newly staged file, unstage leaves it untracked. Plain `git restore <path>` can discard unstaged edits; keep that demonstration on disposable content.

Adding only `notes.txt` does not clear a README change staged earlier. Review the entire staged diff. An empty `git diff` can coexist with staged and untracked changes; inspect status for the full picture.

<details>
<summary>Self-study answer check — open after predicting and observing</summary>

The original sequence finishes with working C, index B, and HEAD B: commit records the selected B snapshot. If you add C again before the first commit, the three views contain C after committing. The decisive condition is the last add, not the last editor save. After the original B commit, adding and then unstaging C restores index B while preserving working C.

</details>

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

Your hands-on contribution starts here. Return to your DL-Example clone in PowerShell; the staging lab is separate. First check the repository, branch, and current work:

```powershell
git rev-parse --show-toplevel
git branch --show-current
git status
```

Choose the path before running commands:

- **First attempt:** inspect status and begin with a clean working tree and no existing `docs/classroom-note` branch. Run the block below once.
- **Resume this same unfinished task:** skip that block. Already on `docs/classroom-note`? Stay there with your uncommitted work and continue. If switching is needed, inspect status first and ask for help with uncommitted changes; from a clean tree, use `git switch docs/classroom-note`. Keep your existing note and continue from the first unfinished step. If the intended note is already committed, continue with Part 2. If a PR already exists, open it and address feedback on the same branch. Preserve existing edits and ask the instructor if they prevent switching.
- **A different new task:** update main and create a different task branch name. Switching to an old branch resumes its existing history; it does not restart it from updated main.

**Run one command at a time. If a command fails, stop and show the message to the instructor.** A clean working tree does not guarantee that a fast-forward update will succeed.

First-attempt commands:

```powershell
git status
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
git switch -c docs/classroom-note
```

Fetch downloads commits and updates `upstream/main`; working files stay unchanged. The fast-forward merge updates local main when history permits; push updates the fork's main. If any command fails, read its message and inspect status with the instructor before continuing.

The student workflow is: **start or resume > edit/save > inspect/add/verify > commit > push > verify the fork's task branch > prepare PR > review diff > create PR > record result**. Maintainer review and merge can follow after class.

### Create a new documentation file

Create `docs/classroom-notes/YOUR-USERNAME.md` in your editor, making the directory if needed. Replace `YOUR-USERNAME` with your GitHub username in both the filename and the `git add` command. Start with these headings:

```markdown
## Save and commit

## Commit and push
```

Write one sentence under each heading. Useful terms: working file, index, local history, fork. Discuss with a partner in a helpful language, then write the final explanation in English. The Git check concerns which location each action changes and which locations remain unchanged; English fluency is not the learning target. Include no personal contact information or training artifacts.

This file starts **untracked**. It differs from the tracked `notes.txt` staging demonstration above. Inspect and select it:

```powershell
git status
git diff
git add docs/classroom-notes/YOUR-USERNAME.md
git diff --staged
```

Status lists a new untracked file, but `git diff` does not show its content. Read it in your editor first. After add, read the staged diff and verify the intended file and text.

**Pause before commit: What will your next commit contain?** Point to the selected file and lines. Remember that a staged diff shows changes from HEAD, not the entire snapshot; unchanged tracked contents remain included.

```powershell
git commit -m "docs: explain local and shared version records"
git status
```

This completes the local part. Your commit has not yet been published. If a repeated attempt reports `nothing to commit`, inspect `git status`, `git log --oneline -3`, and your existing note and commit with the instructor. Continue from the first unfinished step instead of manufacturing another edit.

## Hands-on Part 2: GitHub collaboration

In the same clone and on the same task branch, use PowerShell to publish the branch:

```powershell
git push -u origin docs/classroom-note
```

Keep any authentication error message for diagnosis. After a successful push, open **your fork** on GitHub. Use the branch selector to choose **docs/classroom-note**, then open your new file and verify its text. A fork opened on main will not show the task branch's new file. Continue to the PR comparison after this check.

**Pause after push: Where does your committed change exist now?** Name every location containing it. Explain whether the original main branch has changed and what must happen before it does.

### Prepare a pull request

On the original repository, choose **Pull requests > New pull request**, then **Compare across forks** when needed:

| Question | Selection |
|---|---|
| Where should the change go? | **Base:** `YI-TING-EE13/DL-Example`, branch `main` |
| Where does the change come from? | **Head:** your fork, branch `docs/classroom-note` |

### Review the diff, then create the PR

Check that the diff contains only your intended file and changes. Select **Create pull request**, add a short title and description of what changed and how you checked it, then submit the PR. A reviewer checks the sentences for conceptual accuracy; a synthetic Python check is not required for this task.

### Record and revise your result

In PowerShell, read the current branch and full commit identifier:

```powershell
git branch --show-current
git rev-parse HEAD
```

After creating the PR, keep the branch, full contribution SHA, and PR URL in a temporary note. Then everyone answers the two questions about their own contribution. Edit the PR description **once**, saving all five fields together:

```markdown
## Classroom evidence

- Task branch:
- Contribution commit SHA:
- PR URL:

### Content answer
What did your recorded commit contain, and why?

### Location answer
Where did your committed change exist after push?
```

Show the saved description and diff to the instructor. Keep the self URL in this portable record for follow-up and private offline collection. The two answers are separate from the two sentences in your classroom-note file; no extra commit is needed to submit them. Identify the actual SHA you explain. Later feedback may need another update: commit and push revisions to the same task branch, then update the affected evidence.

An open PR completes publication. All five saved, accessible fields establish receipt; conceptual correctness is checked separately against the actual contribution. Creating a PR does not merge it. Maintainer review, approval, merge, and fork CI permission are separate events and may follow after class.

If publishing is blocked, privately hand in `submission-record.md` with the same five fields, factual pending reasons, and the local commit/error. Use the instructor's agreed collection route; leaving it only on your laptop is not collection. This is pending publication, not a completed PR.

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
