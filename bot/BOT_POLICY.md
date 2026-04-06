<!-- ⚠️ DO NOT COMPRESS, SUMMARIZE, OR SKIP ANY PART OF THIS FILE ⚠️ -->

# Bot Policy — xPyD-plan

## Identity

- **Project:** xPyD-plan
- **Repo:** `xPyD-hub/xPyD-plan`
- **Architecture:** Offline planning tool — analyze vLLM/SGLang/TRT-LLM benchmark data to recommend optimal P:D instance ratios.

## Accounts

| Role | GitHub Account |
|------|---------------|
| Implementer | `hlin99` |
| Reviewer 1 | `hlin99-Review-Bot` |
| Reviewer 2 | `hlin99-Review-BotX` |

## Rules

1. **Never push directly to `main`.** All changes go through PRs.
2. **Rebase onto latest `main`** before every push.
3. **Run pre-commit** before every commit.
4. **Never self-merge.** Wait for both reviewer bots to approve.
5. **All code, docs, issues, PRs in English.**
6. **Conventional Commits** format for all commit messages.
7. **CI must be green** before merge.

## References

- `bot/DESIGN_PRINCIPLES.md` — what to build and why
- `bot/DEV_LOOP.md` — how to iterate
- `bot/REVIEW_POLICY.md` — review process
- `ROADMAP.md` — milestone tracker
