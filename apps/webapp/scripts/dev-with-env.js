// Runs `next dev` against one of the webapp's env files (.env.localhost / .env.remote / .env.prod),
// restarting the dev server whenever that file changes.
//
// Next only watches the `.env*` names it owns (`.env`, `.env.local`, `.env.development`), and the
// values it finds there lose to anything already in process.env — so an edit to `.env.prod` stays
// invisible until the server is killed and started by hand. This wrapper does that automatically.

const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const dotenv = require('dotenv');

const WEBAPP_ROOT = path.resolve(__dirname, '..');
const RESTART_DEBOUNCE_MS = 300;
const SIGKILL_AFTER_MS = 5000;

const envFileName = process.argv[2];
if (!envFileName) {
  console.error('Usage: node scripts/dev-with-env.js <env-file>   (e.g. .env.prod)');
  process.exit(1);
}
const envFilePath = path.join(WEBAPP_ROOT, envFileName);

function log(message) {
  console.log(`\n[${envFileName}] ${message}\n`);
}

function binary(name) {
  const bin = path.join(WEBAPP_ROOT, 'node_modules', '.bin', name);
  if (!fs.existsSync(bin)) {
    console.error(`Could not find ${name} in node_modules/.bin. Run 'npm install' in apps/webapp.`);
    process.exit(1);
  }
  return bin;
}

function readEnvFile() {
  try {
    return dotenv.parse(fs.readFileSync(envFilePath));
  } catch (e) {
    console.error(`Could not read ${envFilePath}: ${e.message}`);
    return null;
  }
}

// The file wins over the surrounding environment, which is what env-cmd did before this script
// existed: `make webapp-dev` exports the repo-root .env first and expects this file to layer on top.
function childEnv() {
  return { ...process.env, ...fileEnv };
}

// Names only. Values are secrets and this goes to a terminal that gets pasted around.
function changedKeys(before, after) {
  const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
  return [...keys].filter((key) => before[key] !== after[key]);
}

let fileEnv = readEnvFile();
if (!fileEnv) {
  process.exit(1);
}

const generate = spawnSync(binary('prisma'), ['generate'], { cwd: WEBAPP_ROOT, stdio: 'inherit', env: childEnv() });
if (generate.status !== 0) {
  process.exit(generate.status ?? 1);
}

const nextBin = binary('next');
let child = null;
let restartTimer = null;
let restartQueue = Promise.resolve();
let shuttingDown = false;

function killGroup(proc, signal) {
  try {
    process.kill(-proc.pid, signal);
  } catch {
    // Already gone.
  }
}

function startNext() {
  // detached puts the dev server and its compile workers in their own process group, so a restart
  // can take all of them down at once rather than orphaning workers that still hold port 3000.
  const running = spawn(nextBin, ['dev', '--turbopack'], {
    cwd: WEBAPP_ROOT,
    stdio: 'inherit',
    env: childEnv(),
    detached: true,
  });
  child = running;
  running.on('exit', (code) => {
    // A deliberate stop clears `child` first, so reaching here means the dev server exited on its own.
    if (child === running) {
      process.exit(code ?? 1);
    }
  });
}

function stopNext() {
  const running = child;
  child = null;
  if (!running || running.exitCode !== null) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const force = setTimeout(() => killGroup(running, 'SIGKILL'), SIGKILL_AFTER_MS);
    running.once('exit', () => {
      clearTimeout(force);
      resolve();
    });
    killGroup(running, 'SIGTERM');
  });
}

async function restart() {
  const updated = readEnvFile();
  if (shuttingDown || !updated) {
    return;
  }
  const changed = changedKeys(fileEnv, updated);
  if (changed.length === 0) {
    return;
  }
  fileEnv = updated;
  log(`changed: ${changed.join(', ')} — restarting the dev server`);
  await stopNext();
  if (!shuttingDown) {
    startNext();
  }
}

// Watching the directory rather than the file itself, because editors that save by writing a temp
// file and renaming it over the original leave an fs.watch on the path pointed at the old inode.
fs.watch(WEBAPP_ROOT, (_event, filename) => {
  if (filename !== envFileName) {
    return;
  }
  clearTimeout(restartTimer);
  // Queued rather than called directly, so that a second save while the server is still coming
  // down cannot start a second dev server alongside the first.
  restartTimer = setTimeout(() => {
    restartQueue = restartQueue.then(restart);
  }, RESTART_DEBOUNCE_MS);
});

['SIGINT', 'SIGTERM'].forEach((signal) => {
  process.on(signal, () => {
    shuttingDown = true;
    clearTimeout(restartTimer);
    stopNext().then(() => process.exit(0));
  });
});

log('watching for changes; the dev server restarts when it is edited');
startNext();
