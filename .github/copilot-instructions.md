# Copilot instructions

This repository keeps all agent instructions in [`AGENTS.md`](../AGENTS.md) at the root, so that no
rule is visible to one coding agent and invisible to another. Read it before making changes.

Subdirectories with their own conventions carry a nested `AGENTS.md` — currently
[`apps/webapp/AGENTS.md`](../apps/webapp/AGENTS.md). Read the one nearest the code you are editing,
in addition to the root file.

Do not add project rules to this file; add them to the nearest `AGENTS.md` instead.
