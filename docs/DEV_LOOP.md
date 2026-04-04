# Autonomous Development Loop Specification

## Loop Steps (Per Iteration)

### Step 1: Sync & Assess
1. Pull latest code
2. Read ROADMAP.md to confirm current progress
3. Read docs/DESIGN_PRINCIPLES.md to confirm design principles
4. Check open issues and PRs: handle any unmerged PRs first

### Step 2: Think & Plan
1. Analyze current code deficiencies and identify the most valuable next improvement
2. Think independently — do not mechanically apply reference implementations
3. Create a GitHub Issue (English) containing:
   - Problem analysis: why this needs to be done
   - Solution design: how to do it and why this approach
   - Acceptance criteria: what "done" looks like
   - Test cases: what tests are needed

### Step 3: Implement
1. Create a branch
2. Write code + tests
3. Ensure ruff check and isort --check pass
4. Commit messages in English, conventional commits format

### Step 4: Self-Review
1. Check each acceptance criterion from the Issue
2. Check code quality: type annotations, docstrings, naming, edge cases
3. Check test coverage: critical paths, boundary values, error scenarios
4. Check for violations of DESIGN_PRINCIPLES.md
5. Fix issues and re-review

### Step 5: Merge & Record
1. Create PR (English, body contains Closes #N)
2. Wait for CI to pass (all checks must be green)
3. Squash merge only after CI passes
4. Update ROADMAP.md to mark progress
5. Push to main

### Step 6: Reflect
1. How much did this iteration advance the final goal?
2. Are there previously made design decisions that need adjustment?
3. If so, update ROADMAP.md or DESIGN_PRINCIPLES.md

## CI Requirements
- All PRs must pass CI before merging (lint + tests on Python 3.10/3.11/3.12)
- If CI fails, fix the code and push again — never merge with failing CI
- If CI is stuck or flaky, investigate and fix the CI itself

## Reporting
- Every 5 iterations, send a summary to the user
- Content: what was accomplished, current progress, challenges, decisions made, next steps
- Report major design choices to the user, but don't wait for reply

## Quality Red Lines
- Never skip tests
- Never merge with failing lint or CI
- Never copy code from other projects
- Every technical decision must have a reason

## Infinite Loop Mechanism

The roadmap milestones are not the finish line. When all existing milestones are complete, the loop does not stop — it enters a **continuous evolution phase**:

### Phase 1: Roadmap-Driven (when milestones exist)
Follow ROADMAP.md milestone order.

### Phase 2: Continuous Evolution (when all milestones are done)
1. **Review the big picture**: Re-examine the entire project:
   - What limitations does the current approach have?
   - How can user experience be improved?
   - How can algorithm accuracy be enhanced?
   - Are there new scenarios to support?
   - Is there code worth refactoring?
   - Are there test coverage gaps?
   - Is documentation complete?

2. **Create new milestones**: Organize discovered improvements into new milestones, update ROADMAP.md

3. **Continue the loop**: Return to Phase 1

### Sources of Inspiration for Continuous Evolution
- Run the project's own tests, look for uncovered edge cases
- Validate recommendations with real vLLM bench data (if public datasets available)
- Compare with similar industry tools, find differentiation opportunities
- Think about complex scenarios: multi-model co-location, heterogeneous GPUs, dynamic loads
- Performance optimization: computational efficiency at scale
- Observability: better logging, metrics, debugging info
