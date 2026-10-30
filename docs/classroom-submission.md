# Classroom submission: five evidence fields

These answers concern your actual contribution. They are separate from the two sentences in your classroom-note file; no extra commit is needed just to submit them.

1. Gather your actual task branch, contribution commit SHA, and PR URL temporarily. On the task branch, use `git branch --show-current` and `git rev-parse HEAD`; record that SHA only if it identifies your contribution, not an unrelated starting commit.
2. Answer Content and Location about what actually happened. If push is blocked, state current locations and expected locations after a successful push separately.
3. With a PR open, update its description once with all five fields after answering; show the saved page. With publication blocked, privately hand in the same record as `submission-record.md` through the agreed route, including the last completed step, error, and pending reason.

In the companion slides: create the PR at 22, gather at 23, answer/save at 24. Later feedback can require another update; keep the same task branch and identify the actual SHA you explain.

```markdown
## Classroom evidence

- Task branch:
- Contribution commit SHA:
- PR URL:

### Content answer
What did your commit record, and why?

### Location answer
Where is your commit now?
If push is blocked, distinguish now from the expected locations after a successful push.

### Publication note (only if blocked)
Last completed step:
Error and pending reason:
```

| Actual progress | PR URL | Location answer |
|---|---|---|
| PR exists | Its actual URL | Explain observed locations using the actual commit and PR state. |
| Commit exists; push is blocked | `N/A - publication pending` if no PR exists | State where the commit is now, then where it would exist after a successful push. Do not describe an expected push as completed. |
| Push succeeded; PR creation is blocked | `N/A - publication pending` | State the actual post-push locations. The remaining pending step is PR creation. |

Keep an existing PR URL if one already exists. If the contribution has not been committed, use `N/A - commit pending` for its SHA and describe the staged/working state; do not invent a commit or use an unrelated HEAD as evidence. This remains incomplete contribution evidence for follow-up.

An open PR completes publication. Five saved, accessible fields establish receipt; correctness is separate. Maintainer approval, merge, and CI permission are separate from collection. A record left only on your computer is not collected; pending publication is not a completed PR. Omit credentials and personal contact details from the contribution.
