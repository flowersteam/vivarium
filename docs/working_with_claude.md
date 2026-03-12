# Working with Claude Code on this project

_Internal guide — not user-facing._

---

## Starting a new session

Always open with a short context-setting message:

> "We're working on `PLAN.md`, currently on phase X / task Y.
> Today I want to: [specific goal].
> Since last session: [anything you changed, or 'nothing changed']."

This lets Claude immediately read `PLAN.md` and orient without guessing.
If the goal for the session is unclear, Claude will ask before starting work.

---

## When you change code yourself

**Before** telling Claude to continue: commit your changes (even as a WIP commit), then say:

> "I made the following changes since last session: [brief description]. See `git log`."

A committed state means Claude can run `git diff` or `git log` to understand exactly what changed.
An uncommitted working tree is ambiguous. If you can't commit, describe the changes explicitly.

---

## Resuming an aborted session

If a session ends mid-task (no clean completion), before closing say:

> "We need to stop. We were in the middle of [task]. Status: [what's done, what remains]."

Claude will update `PLAN.md` before closing. Next session, use the standard opening template.

---

## Keeping PLAN.md accurate

- Claude updates it at the end of every completed task and at session end (noting in-progress work)
- You can edit it directly between sessions — if you do, mention it when resuming
- If the plan turns out to be wrong mid-session, update it immediately and continue

---

## Keeping CLAUDE.md accurate

- Updating `CLAUDE.md` is the **last step** of any refactoring task
- Either Claude does it, or you do and mention it at the next session start
- If you notice an inaccuracy, correct it and commit — no need to wait for a session

---

## Task scoping

Each session should have **one clearly scoped task** with a clear done condition:

- "All dead code removed and tests pass"
- "README rewritten and reviewed"
- "Session 3 notebook validated against current API"

Avoid "let's do several things." If a session runs long, split the task and update the plan.

---

## Agent usage

Claude uses built-in agents automatically when appropriate:

- **Explore agent**: broad codebase exploration (at the start of audit or refactor tasks)
- **Plan agent**: designing significant refactors before implementing

No configuration needed. Claude will say when an agent is being used.
Custom skills or hooks are not used for this project.

---

## Git workflow

- One branch per significant change (refactor, feature, documentation pass)
- Merged back to the working branch when done and reviewed
- Start each session with a clean working tree when possible
- Run `pytest` before and after any structural change
