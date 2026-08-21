#!/usr/bin/env node
// Point this checkout's git hooks at .githooks/, the same thing `make githooks-install` does.
//
// The webapp's `prepare` script runs this after every `npm install`, so a webapp contributor gets
// the hooks without having to know they exist. Python contributors run the make target instead,
// since they may never install node.
//
// Node rather than shell because npm runs `prepare` through cmd.exe on Windows, where the make
// target and the one-line `git config` it wraps are both unavailable.
//
// Everything here is best-effort -- a failure to install a convenience hook must never fail an
// install -- which is also why `prepare` ends in `|| exit 0`: a checkout holding apps/webapp alone
// has no ../../.githooks to find. It refuses to overwrite a core.hooksPath someone else set, so a
// personal husky or hand-written hook keeps working.

const { execFileSync } = require('node:child_process');
const { existsSync } = require('node:fs');
const { join } = require('node:path');

const HOOKS_PATH = '.githooks';

function git(...args) {
  return execFileSync('git', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
}

try {
  // CI checks out fresh and runs its own gates, so installing hooks there is noise at best.
  if (process.env.CI || process.env.SKIP_GITHOOKS) {
    process.exit(0);
  }

  const repoRoot = git('rev-parse', '--show-toplevel');
  // A tarball or a vendored copy has no hooks to install into.
  if (!existsSync(join(repoRoot, HOOKS_PATH, 'pre-commit'))) {
    process.exit(0);
  }

  let current = '';
  try {
    current = git('config', '--get', 'core.hooksPath');
  } catch {
    // Unset: `git config --get` exits 1, which execFileSync turns into a throw.
  }

  if (current === HOOKS_PATH) {
    process.exit(0);
  }
  if (current) {
    console.log(`git hooks: leaving core.hooksPath as '${current}'. Run 'make githooks-install' to use ${HOOKS_PATH}/.`);
    process.exit(0);
  }

  git('config', 'core.hooksPath', HOOKS_PATH);
  console.log(`git hooks: enabled ${HOOKS_PATH}/ for this checkout. Undo with 'make githooks-uninstall'.`);
} catch {
  // No git, no repo, a read-only config: none of that is worth failing an install over.
  process.exit(0);
}
