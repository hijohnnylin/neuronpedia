# Contributing

## Issues

Use [Github Issues](https://github.com/hijohnnylin/neuronpedia/issues) for the following:

- Feature Requests
  - This can be as simple as one sentence with a link to a paper you want to integrate, and stating its importance/relevance
  - More details will increase the odds of it being prioritized
- Bug Reports
  - Please include steps to reproduce the bug
- Questions
  - Shorter questions can be answered in our [Slack #neuronpedia](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-3z9o0hxjl-MDX9pbATO2qESOazNDLpdQ)
  - Sensitive questions can be sent [via email](mailto:johnny@neuronpedia.org)

## Pull Requests

Submit a pull request to get your changes merged into Neuronpedia.

We will work with you to get your PR in, including reviewing and helping you modify your code.

Please adhere to the following:

- In the pull request **code**:
  - Check that you are not committing secrets/passwords
  - Ensure it does not break existing APIs
  - Add documentation where necessary - feel free to modify the READMEs, they are not off-limits.
- In the pull request **text**:
  - Summarize the problem you are solving
  - Use bullet points if there are multiple changes being made
  - Describe how you tested your changes
  - Highlight any limitations and TODOs (and create new issues for them)

## Checks Before You Commit

CI runs a few gates that are quick to fail and quick to fix: `ruff check` and `ruff format` for the
five Python services, `eslint`, `prettier` and `tsc` for the webapp, and some small scripts that
read committed files (the OpenAPI specs, the shared lint config, the agent instruction files).

You can run the fast ones automatically before each commit. `.githooks/pre-commit` checks only the
files the commit touches, and only with toolchains you have installed - a webapp-only checkout is
never asked for `uv`, and vice versa. It changes nothing in your working tree; when something
fails, it prints the command that fixes it.

Enabling it is per checkout and optional:

```
make githooks-install     # python contributors
make githooks-uninstall   # turn it back off
```

Webapp contributors get it from `npm install`, which runs `.githooks/install.js` through the
`prepare` script. Both do the same one thing: point `core.hooksPath` at `.githooks/`. That setting
lives in `.git/config`, which is not tracked, so it affects nobody else. The installer leaves a
`core.hooksPath` you set yourself alone.

To skip the checks for one commit, use `git commit --no-verify`, or set `SKIP_GITHOOKS=1`.

The slow gates stay out of the hook, so run these yourself before opening a pull request:

| Check                              | Command                                     |
| ---------------------------------- | ------------------------------------------- |
| Python lint and format, every app  | `make python-lint` (`make python-lint-fix`) |
| Python types                       | `cd apps/<app> && uv run pyright .`         |
| Webapp lint **and** `tsc --noEmit` | `cd apps/webapp && npm run lint`            |
| Webapp unit tests                  | `cd apps/webapp && npm test`                |
| Python tests                       | `cd apps/<app> && make test`                |

## AI Coding / Agents

- Code written or assisted by AI and/or AI agents is welcome, however, we ask that you (the human) manually review all code and verify correct functionality first.
- We've also added some [Cursor Rules](https://docs.cursor.com/context/rules-for-ai) files, though they are not extensive

## Contributing Terms

Neuronpedia is licensed under the [Apache License, Version 2.0](./LICENSE). There is no
separate agreement to sign: under section 5 of that license, anything you deliberately
submit for inclusion in the project is contributed under the same Apache 2.0 terms,
unless you say otherwise in the pull request.

You keep the copyright to your work. Apache 2.0 also includes an express patent grant
from each contributor, which is one of the reasons the project uses it.

If a contribution contains code you did not write, say so in the pull request and name
its source and license, so it can be attributed properly in `NOTICE`.

## Code of Conduct

All contributions should follow our [code of conduct](CODE_OF_CONDUCT.md).
