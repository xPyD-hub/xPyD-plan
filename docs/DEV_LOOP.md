# Development Loop

Autonomous infinite loop. Runs until explicitly stopped.

## Setup (every iteration)
```
git config user.email "tony.lin@intel.com"
git config user.name "hlin99"
```

## Each Iteration

1. Pull latest code
2. Read `ROADMAP.md` — find the next incomplete milestone
3. Read `DESIGN_PRINCIPLES.md` — follow the rules
4. Check open issues/PRs — handle unmerged PRs first (fix CI failures, address review comments)
5. If no milestone left, create new ones (see Phase 2 below)
6. Create GitHub Issue: problem, solution, acceptance criteria, tests
7. Create branch, implement code + tests
8. Pass lint: `ruff check src tests && isort --check src tests`
9. Update `docs/iterations/current.md` with what you did this iteration
10. Create PR (body contains `Closes #N`)
11. Wait for CI green. Fix failures. Never merge red CI.
12. **Wait for reviewer bots** — do NOT self-merge. Two reviewer bots (`hlin99-Review-Bot` and `hlin99-Review-BotX`) will be auto-assigned.
13. Handle review result:
    - **2 approvals** → auto-merge → update ROADMAP.md → go to step 1
    - **request changes** → fix code, push to same PR → wait for re-review (max 10 min to fix)
    - **closed by reviewer** → iteration failed → push update to `docs/iterations/current.md` on main recording the failure (what was attempted, why rejected, reviewer comments) → go to step 1 with a different task
14. Go to step 1

## Review Rules (see REVIEW_POLICY.md)

- 2 reviewer bots are auto-assigned on PR creation
- Either reviewer can close the PR (idea rejected) — one close = PR dead
- Both must approve for merge
- Reviewer timeout: 15 minutes → PR auto-closed
- Total round timeout: 1 hour → PR auto-closed
- Implementer (hlin99) must NEVER approve or merge their own PR

## Timing

| Parameter | Value |
|-----------|-------|
| Iteration interval | 10 minutes |
| PR wait for review | max 15 minutes |
| Fix after request changes | max 10 minutes |
| Total round timeout | 1 hour |

## Deliverables (every iteration)

Every PR MUST include:
- Code changes (if any)
- Tests for new code
- Updated `docs/iterations/current.md` describing what was done

## Rules
- Committer must be `hlin99 <tony.lin@intel.com>` — always set git config before any commit
- All code, docs, issues, PRs in English
- Commit messages: conventional commits format
- Never self-merge — wait for reviewer bots

## Phase 1: Roadmap-Driven
Follow ROADMAP.md milestones in order.

## Phase 2: Continuous Evolution
When all milestones are done:
1. Review the project — find limitations, improvements, new scenarios
2. Create new milestones in ROADMAP.md
3. Return to Phase 1

## Iteration Tracking

`docs/iterations/current.md` must maintain a running log at the bottom:

```markdown
## Iteration History

| # | Date | Task | Result | Reviewer Comments |
|---|------|------|--------|-------------------|
| 1 | 2026-04-06 | Added X feature | ✅ merged | Both approved |
| 2 | 2026-04-06 | Refactored Y | ❌ closed | BotX: idea not valuable |
| 3 | 2026-04-06 | Fixed Z bug | ✅ merged | Bot requested changes, fixed |
```

This table is the source of truth for iteration success/failure rate.
