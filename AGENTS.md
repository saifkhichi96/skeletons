# AGENTS.md

## Philosophy

- **Keep it simple.** Prefer the smallest correct solution. Avoid unnecessary abstractions, defensive code, and speculative features.
- **Understand before changing.** Read the relevant code, understand existing conventions, and identify the root cause before making edits.
- **Minimal diffs.** Change only what is necessary. Do not refactor unrelated code, reformat entire files, or introduce dependencies without justification.
- **No shortcuts.** Fix underlying problems rather than hiding symptoms. Do not introduce hacks or silent fallbacks.
- **Verify your work.** For significant features, do not assume a change works. Test it, inspect the diff, and address failures before declaring completion. Static verification is enough for trivial changes.

## Project Conventions

```
src/
├── docs/               # Sphinx docs
├── skeletons/          # core library
├── rigs/               # skeleton definitions
│   └── playground_app/ # optional GUI
├── examples/           # usage examples
└── tests/
```

- Keep dependencies minimal. Core package uses only `torch` and `numpy`.
- Follow SOLID and `ruff.toml`. Keep existing architecture, naming conventions.
- Use PEP 8, type hints, and NumPy-style docstrings for public APIs.
- Export only public APIs through `src/skeletons/__init__.py`.
- Update relevant Sphinx API references, tutorials, and examples when introducing public functionality.

## Workflow

1. **Inspect:** Read relevant files and tests. Understand behavior and dependencies before editing.
2. **Implement:** Make focused changes consistent with existing patterns. Use `apply_patch` for manual edits.
3. **Test:** Add or update deterministic `pytest` tests for behavioral changes, including edge cases for non-trivial changes.
4. **Validate:** Run relevant tests, and Ruff checks. Report failures.
5. **Review:** Inspect the final diff for unintended changes, complexity, and regressions.
6. **Summarize:** Explain what changed, why, and how it was verified. Keep summary concise.

## Git & Documentation

- Use `feat/<description>` or `fix/<description>` branches based on `dev`.
- Follow Conventional Commits.
- Update `CHANGELOG.md` for meaningful changes.
- Do not commit, merge, or push unless explicitly requested.
- PRs require a clear description, reviewer, and passing CI.

## Agent Behavior

- **Be autonomous, not reckless.** Make reasonable implementation decisions without requesting approval for every detail. Ask when requirements are genuinely ambiguous or decisions have significant consequences.
- **Do not guess.** Inspect code, documentation, or tests when uncertain. State assumptions explicitly.
- **Do not overengineer.** A few clear lines are better than an elaborate abstraction used once.
- **Preserve existing behavior** unless changing it is the explicit objective.
- **Do not fabricate success.** If something cannot be implemented or verified, explain what remains.
- **Leave the codebase better, not bigger.** Prioritize readability, maintainability, and correctness over cleverness.