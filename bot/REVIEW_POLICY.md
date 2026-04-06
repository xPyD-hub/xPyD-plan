<!-- ⚠️ DO NOT COMPRESS, SUMMARIZE, OR SKIP ANY PART OF THIS FILE ⚠️ -->

# Review Policy

## Roles

| Role | GitHub Account |
|------|---------------|
| Implementer | `hlin99` |
| Reviewer 1 | `hlin99-Review-Bot` |
| Reviewer 2 | `hlin99-Review-BotX` |

## Review Criteria

### 1. Idea Value
- Does the direction align with project goals?
- **If NO → close PR immediately** (one close = PR rejected)

### 2. Code Quality
- Correct code, tests included/passing
- `bot/iterations/current.md` updated
- **If idea good but code has issues → request changes**

## Decision Rules

| Scenario | Action |
|----------|--------|
| Both approve | Auto-merge |
| One approves, one requests changes | Fix, re-review |
| Either closes | PR closed, iteration failed |
| Timeout (15 min no review) | PR closed |
| Total timeout (1 hour) | PR closed |

## Auto-Merge Requirements

- 2 approvals from designated reviewers
- CI green
- No unresolved review comments
