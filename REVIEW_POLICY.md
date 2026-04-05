# Review Policy

## Roles

| Role | GitHub Account | Action |
|------|---------------|--------|
| Implementer | `hlin99` | Write code, submit PRs, fix issues |
| Reviewer 1 | `hlin99-Review-Bot` | Review PRs: approve / request changes / close |
| Reviewer 2 | `hlin99-Review-BotX` | Review PRs: approve / request changes / close |

## Timing

| Parameter | Value |
|-----------|-------|
| Iteration interval | 10 minutes |
| PR wait for review | max 15 minutes |
| Fix after request changes | max 10 minutes |
| Reviewer check frequency | every 5 minutes |
| Reviewer response deadline | 15 minutes after assign |
| Reviewer timeout action | close PR (iteration failed) |
| Total round timeout | 1 hour from PR creation |
| Round timeout action | close PR (iteration failed) |

## Review Criteria

Reviewers evaluate each PR on two dimensions:

### 1. Idea Value
- Is the direction/approach valuable for the project?
- Does it align with the project goals?
- **If NO → close PR immediately** (one close = PR rejected)

### 2. Code Quality
- Is the code correct?
- Are tests included/passing?
- Is `docs/iterations/current.md` updated with clear description?
- Does `docs/guide.md` reflect changes (if applicable)?
- **If idea is good but code has issues → request changes**

## Decision Rules

| Scenario | Action |
|----------|--------|
| Both reviewers approve | Auto-merge |
| One approves, one requests changes | Implementer fixes, reviewers re-review |
| Either reviewer closes | PR closed, iteration failed |
| Both approve after fixes | Auto-merge |
| Timeout (15min no review) | PR closed, iteration failed |
| Total timeout (1 hour) | PR closed, iteration failed |

## Iteration Record

Every PR MUST update `docs/iterations/current.md` with:
- What was done this iteration
- Result: merged / closed (with reason)
- Reviewer scores/comments summary

## Auto-Merge Requirements

- 2 approvals from designated reviewers
- CI passes (all checks green)
- No unresolved review comments
