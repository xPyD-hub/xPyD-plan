<!-- ⚠️ DO NOT COMPRESS, SUMMARIZE, OR SKIP ANY PART OF THIS FILE ⚠️ -->

# Development Loop

Autonomous infinite loop. Runs until explicitly stopped.

## Each Iteration

1. Pull latest `main`, rebase branch
2. Read `ROADMAP.md` — find next incomplete milestone
3. Read `bot/DESIGN_PRINCIPLES.md` — follow the rules
4. Check open issues/PRs — handle unmerged PRs first
5. Create GitHub Issue: problem, solution, acceptance criteria, tests
6. Create branch, implement code + tests
7. Lint: `ruff check xpyd_plan tests`
8. Update `bot/iterations/current.md`
9. Create PR (body contains `Closes #N`)
10. Wait for CI green. Fix failures.
11. Wait for reviewer bots (see `bot/REVIEW_POLICY.md`)
12. Handle review result:
    - **2 approvals** → auto-merge → update ROADMAP → step 1
    - **request changes** → fix, push → wait for re-review
    - **closed** → record failure in `bot/iterations/current.md` → step 1

## Deliverables (every iteration)

- Code changes + tests
- Updated `bot/iterations/current.md`
