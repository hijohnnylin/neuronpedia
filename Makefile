# Neuronpedia development commands.
#
# Every service runs directly on your machine. Prerequisites:
#
#   Node.js 22+  ->  make install-nodejs
#   Postgres 16+ with the pgvector extension  ->  see README "Set Up Your Database"
#   uv           ->  https://docs.astral.sh/uv/getting-started/installation/
#
# Commands follow the pattern 'make [app]-[action]', e.g. 'make webapp-dev'.

SHELL := /bin/bash

# Shared secret the webapp presents to the local inference/autointerp/graph
# servers. Must match INFERENCE_SERVER_SECRET (and friends) in
# apps/webapp/.env.localhost.
LOCALHOST_SECRET ?= localhost-secret

# The repo-root .env holds your personal keys (OPENAI_API_KEY, HF_TOKEN) and is
# layered under each service's own env file. Created by `make init-env`.
LOAD_ROOT_ENV = set -a; if [ -f .env ]; then . ./.env; fi; set +a;

.PHONY: help init-env install-nodejs \
	db-init db-check db-status db-reset \
	host-add host-list host-remove \
	webapp-install webapp-dev webapp-build webapp-run webapp-openapi sdk-dry-run \
	inference-install inference-dev inference-list-configs inference-openapi \
	autointerp-install autointerp-dev autointerp-openapi \
	graph-install graph-dev graph-openapi \
	sparsity-install sparsity-dev sparsity-openapi \
	nla-install nla-dev nla-openapi \
	engine-link engine-unlink engine-status \
	githooks-install githooks-uninstall \
	python-lint python-lint-fix agent-rules-check openapi openapi-check

# Every Python service under apps/. Used by the lint targets at the bottom of this file, which
# mirror .github/workflows/python-lint.yml.
PYTHON_APPS = autointerp graph inference nla sparsity

help: ## Show available commands
	@echo -e "\n\033[1;35mCommands follow the pattern 'make [app]-[action]'.\nFor example, 'make webapp-dev' runs the webapp in development mode.\033[0m"
	@awk 'BEGIN {FS = ":.*## "; printf "\n"} /^[a-zA-Z_-]+:.*## / { printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init-env: ## Create the repo-root .env with your personal API keys
	@echo "Initializing environment..."
	@if [ -f .env ]; then \
		read -p "'.env' file already exists. Do you want to overwrite it? (y/N) " confirm; \
		if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
			echo "Aborted."; \
			exit 1; \
		fi; \
		echo "Clearing existing .env file."; \
	else \
		echo "Creating new .env file."; \
	fi; \
	echo "" > .env
	@read -p "Enter your OpenAI API key - this is optional, but it is required for Search Explanations to work (press Enter to skip): " api_key; \
	if [ ! -z "$$api_key" ]; then \
		echo "OPENAI_API_KEY=$$api_key" >> .env; \
		echo "OpenAI API key added to .env"; \
	else \
		echo "No API key provided. The Search Explanations feature will not work."; \
	fi
	@read -p "Enter your Hugging Face token - this is optional, but it is required for access to gated HuggingFace models (press Enter to skip): " hf_token; \
	if [ ! -z "$$hf_token" ]; then \
		echo "HF_TOKEN=$$hf_token" >> .env; \
		echo "Hugging Face token added to .env"; \
	else \
		echo "No Hugging Face token provided. Gated models may not be accessible."; \
	fi
	@echo "Environment initialized successfully."

install-nodejs: ## Install Node.js via nvm
	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
	# Need to source NVM in the same shell
	. ${HOME}/.nvm/nvm.sh && nvm install 22

# ----------------------------------------------------------------- database --

db-check: ## Database: verify Neuronpedia can reach Postgres and that pgvector is installed
	@cd apps/webapp && npx --no-install env-cmd -f .env.localhost --use-shell '\
		psql "$$POSTGRES_URL_NON_POOLING" -tAc "select 1" > /dev/null 2>&1 \
			|| { echo "Cannot connect to Postgres at $$POSTGRES_URL_NON_POOLING"; echo "Start Postgres, or edit POSTGRES_* in apps/webapp/.env.localhost."; exit 1; }; \
		psql "$$POSTGRES_URL_NON_POOLING" -tAc "select 1 from pg_available_extensions where name = '"'"'vector'"'"'" \
			| grep -q 1 || { echo "The pgvector extension is not available on this Postgres server."; echo "Install it: https://github.com/pgvector/pgvector#installation"; exit 1; }; \
		echo "Postgres is reachable and pgvector is available."'

# Read-only, and the gate an agent needs before db-init: that target is safe on an empty database
# and destructive-by-surprise on a populated one, since `prisma db seed` rewrites table contents.
# Counts user tables in every non-system schema rather than just `public`, so a POSTGRES_URL with
# ?schema= still reads as populated. Exits non-zero when it finds any, so `make db-status &&
# make db-init` is a safe one-liner.
db-status: ## Database: report whether the schema already exists. Non-zero means already initialized
	@$(MAKE) db-check
	@cd apps/webapp && npx --no-install env-cmd -f .env.localhost --use-shell '\
		tables=$$(psql "$$POSTGRES_URL_NON_POOLING" -tAc "select count(*) from information_schema.tables where table_schema not in ('"'"'pg_catalog'"'"', '"'"'information_schema'"'"')"); \
		if [ "$$tables" = "0" ]; then \
			echo "Database is empty: db-init is safe to run."; \
		else \
			echo "Database already has $$tables tables, so it is already initialized."; \
			echo "db-init would re-seed it, and db-reset would drop it. Leave both to a human."; \
			exit 1; \
		fi'

db-init: ## Database: create the schema, seed it, and apply pgvector tuning
	@$(MAKE) db-check
	@echo "Applying migrations..."
	@cd apps/webapp && npx --no-install env-cmd -f .env.localhost --use-shell 'prisma migrate deploy'
	@echo "Seeding..."
	@cd apps/webapp && npx --no-install env-cmd -f .env.localhost --use-shell 'prisma db seed'
	@echo "Applying pgvector settings..."
	@cd apps/webapp && npx --no-install env-cmd -f .env.localhost --use-shell \
		'psql "$$POSTGRES_URL_NON_POOLING" -q -f prisma/pgvector-init/pgvector.sql'
	@echo "Database ready."

HOST_CLI = cd apps/webapp && npx --no-install env-cmd -f .env.localhost --use-shell \
	'ts-node --compiler-options {\"module\":\"CommonJS\"} scripts/compute-host.ts'

host-add: ## Hosts: register a local GPU server. Required: SERVICE, MODEL, URL. Optional: SOURCES, SOURCE_SETS, NLA_SOURCE, NAME
	@if [ -z "$(SERVICE)" ] || [ -z "$(MODEL)" ] || [ -z "$(URL)" ]; then \
		echo "Error: SERVICE, MODEL and URL are all required."; \
		echo "  e.g. make host-add SERVICE=INFERENCE MODEL=gpt2-small URL=http://127.0.0.1:5002 SOURCES=6-res-jb"; \
		exit 1; \
	fi
	@$(HOST_CLI) add --service $(SERVICE) --model $(MODEL) --url $(URL) \
		$(if $(NAME),--name $(NAME),) \
		$(if $(SOURCES),--sources $(SOURCES),) \
		$(if $(SOURCE_SETS),--source-sets $(SOURCE_SETS),) \
		$(if $(NLA_SOURCE),--nla-source $(NLA_SOURCE),)

host-list: ## Hosts: show registered GPU servers. Optional: SERVICE
	@$(HOST_CLI) list $(if $(SERVICE),--service $(SERVICE),)

host-remove: ## Hosts: deregister a GPU server. Required: SERVICE, URL
	@if [ -z "$(SERVICE)" ] || [ -z "$(URL)" ]; then \
		echo "Error: SERVICE and URL are both required."; \
		exit 1; \
	fi
	@$(HOST_CLI) remove --service $(SERVICE) --url $(URL)

# host-add writes to the local database directly, which a deployed environment
# will not accept from a laptop. These two go through the admin API instead.
REGISTER_CLI = python3 apps/inference/scripts/register_host.py

host-register: ## Hosts: register a GPU server with a deployed webapp over the API. Required: SERVICE, MODEL, URL. Optional: SOURCES, SOURCE_SETS, NLA_SOURCE, NAME, PROVIDER, PROVIDER_REF, WEBAPP, ENVIRONMENT
	@if [ -z "$(SERVICE)" ] || [ -z "$(MODEL)" ] || [ -z "$(URL)" ]; then \
		echo "Error: SERVICE, MODEL and URL are all required."; \
		echo "  e.g. make host-register SERVICE=INFERENCE MODEL=gemma-2-2b \\"; \
		echo "         URL=https://abc123-5002.proxy.runpod.net SOURCE_SETS=gemmascope-res-16k"; \
		echo "  Needs NEURONPEDIA_ADMIN_API_KEY (an admin key, from Settings)."; \
		exit 1; \
	fi
	@$(REGISTER_CLI) register --service $(SERVICE) --model $(MODEL) --url $(URL) \
		$(if $(NAME),--name $(NAME),) \
		$(if $(SOURCES),--sources $(SOURCES),) \
		$(if $(SOURCE_SETS),--source-sets $(SOURCE_SETS),) \
		$(if $(NLA_SOURCE),--nla-source $(NLA_SOURCE),) \
		$(if $(PROVIDER),--provider $(PROVIDER),) \
		$(if $(PROVIDER_REF),--provider-ref $(PROVIDER_REF),) \
		$(if $(WEBAPP),--webapp $(WEBAPP),) \
		$(if $(ENVIRONMENT),--environment $(ENVIRONMENT),)

host-parity: ## Hosts: check the ComputeHost registry routes everything the old tables did. Read-only. Optional: ENV_FILE
	@cd apps/webapp && npx --no-install env-cmd -f $(if $(ENV_FILE),$(ENV_FILE),.env.localhost) --use-shell \
		'ts-node -r tsconfig-paths/register --compiler-options {\"module\":\"CommonJS\"} scripts/compute-host-parity.ts'

host-deregister: ## Hosts: remove a GPU server from a deployed webapp over the API. Required: SERVICE, URL. Optional: WEBAPP
	@if [ -z "$(SERVICE)" ] || [ -z "$(URL)" ]; then \
		echo "Error: SERVICE and URL are both required."; \
		exit 1; \
	fi
	@$(REGISTER_CLI) deregister --service $(SERVICE) --url $(URL) \
		$(if $(WEBAPP),--webapp $(WEBAPP),)

db-reset: ## Database: DROP the schema and rebuild it - this deletes your local data!
	@echo "WARNING: This will delete all data in your local Neuronpedia database!"
	@read -p "Are you sure you want to continue? (y/N) " confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi
	@cd apps/webapp && npx --no-install env-cmd -f .env.localhost --use-shell 'prisma migrate reset --force'
	@$(MAKE) db-init

# ------------------------------------------------------------------- webapp --

webapp-install: ## Webapp: install dependencies
	@if ! command -v npm > /dev/null 2>&1; then \
		echo "Error: npm is not installed. Please install nodejs first with 'make install-nodejs'."; \
		exit 1; \
	fi
	cd apps/webapp && npm install

webapp-dev: ## Webapp: run with hot reload on http://localhost:3000
	$(LOAD_ROOT_ENV) \
	cd apps/webapp && npm run dev:localhost

webapp-build: ## Webapp: production build
	$(LOAD_ROOT_ENV) \
	cd apps/webapp && npm run build:localhost

webapp-run: ## Webapp: serve the production build on http://localhost:3000
	$(LOAD_ROOT_ENV) \
	cd apps/webapp && npm run start:localhost

webapp-openapi: ## Webapp: regenerate lib/api/*.d.ts from the servers' openapi.json
	cd apps/webapp && npm run openapi

sdk-dry-run: ## SDKs: build the published clients from the committed specs without uploading. SERVICE=inference|autointerp
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: SERVICE not specified, e.g. make sdk-dry-run SERVICE=inference"; \
		exit 1; \
	fi
	python3 .github/scripts/publish_sdk.py --service $(SERVICE) --force

# ---------------------------------------------------------------- inference --

inference-install: ## Inference: install dependencies
	cd apps/inference && uv sync $(call engine_relink,inference)

inference-dev: ## Inference: run on port 5002. Required: MODEL_SOURCESET=gpt2-small.res-jb. Options: AUTORELOAD=1
	@if [ -z "$(MODEL_SOURCESET)" ]; then \
		echo "Error: MODEL_SOURCESET not specified. Please specify a model+source configuration, e.g. to load .env.inference.gpt2-small.res-jb, run: make inference-dev MODEL_SOURCESET=gpt2-small.res-jb"; \
		echo "Run 'make inference-list-configs' to see available configurations."; \
		exit 1; \
	fi
	@if [ ! -f ".env.inference.$(MODEL_SOURCESET)" ]; then \
		echo "Error: Configuration file .env.inference.$(MODEL_SOURCESET) not found."; \
		echo "Run 'make inference-list-configs' to see available configurations."; \
		exit 1; \
	fi
	@echo "Using model configuration: .env.inference.$(MODEL_SOURCESET)"
	$(call engine_banner,inference)
	set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	. ./.env.inference.$(MODEL_SOURCESET); \
	SECRET=$(LOCALHOST_SECRET); \
	set +a; \
	cd apps/inference && $(call uv_run,inference) python start.py $(if $(AUTORELOAD),--reload,)

inference-list-configs: ## Inference: list available MODEL_SOURCESET values
	@echo -e "\nAvailable Inference Configurations (.env.inference.*)\n================================================\n"
	@for config in $$(ls .env.inference.*); do \
		name=$$(echo $$config | sed 's/^.env.inference.//'); \
		echo -e "\033[1;36m$$name\033[0m"; \
		model_id=$$(grep "^MODEL_ID=" $$config | cut -d'=' -f2); \
		sae_sets=$$(grep "^SAE_SETS=" $$config | cut -d'=' -f2); \
		echo -e "    Model: \033[33m$$model_id\033[0m"; \
		echo -e "    Source/SAE Sets: \033[32m$$sae_sets\033[0m"; \
		echo -e "    \033[1;35mmake inference-dev MODEL_SOURCESET=$$name\033[0m"; \
		echo ""; \
	done

inference-openapi: ## Inference: rewrite apps/inference/openapi.json from the pydantic models
	cd apps/inference && $(call uv_run,inference) python dump_openapi.py

# --------------------------------------------------------------- autointerp --

autointerp-install: ## Autointerp: install dependencies
	cd apps/autointerp && uv sync

autointerp-dev: ## Autointerp: run on port 5003
	$(LOAD_ROOT_ENV) \
	export SECRET=$(LOCALHOST_SECRET); \
	cd apps/autointerp && uv run python server.py

autointerp-openapi: ## Autointerp: rewrite apps/autointerp/openapi.json from the pydantic models
	cd apps/autointerp && $(call uv_run,autointerp) python dump_openapi.py

# -------------------------------------------------------------------- graph --

graph-install: ## Graph: install dependencies
	cd apps/graph && uv sync $(call engine_relink,graph)

graph-dev: ## Graph: run on port 5004. Options: AUTORELOAD=1
	$(call engine_banner,graph)
	$(LOAD_ROOT_ENV) \
	export SECRET=$(LOCALHOST_SECRET); \
	cd apps/graph && $(call uv_run,graph) python start.py $(if $(AUTORELOAD),--reload,)

graph-openapi: ## Graph: rewrite apps/graph/openapi.json from the pydantic models
	cd apps/graph && $(call uv_run,graph) python dump_openapi.py

# ----------------------------------------------------------------- sparsity --

sparsity-install: ## Sparsity: install dependencies
	cd apps/sparsity && uv sync

sparsity-dev: ## Sparsity: run on port 5005
	$(LOAD_ROOT_ENV) \
	export SECRET=$(LOCALHOST_SECRET); \
	cd apps/sparsity && uv run uvicorn server:app --port 5005

sparsity-openapi: ## Sparsity: rewrite apps/sparsity/openapi.json from the pydantic models
	cd apps/sparsity && $(call uv_run,sparsity) python dump_openapi.py

# ---------------------------------------------------------------------- nla --

nla-install: ## NLA: install dependencies
	cd apps/nla && uv sync $(call engine_relink,nla)

nla-dev: ## NLA: run on port 5009
	$(call engine_banner,nla)
	$(LOAD_ROOT_ENV) \
	export SECRET=$(LOCALHOST_SECRET); \
	cd apps/nla && $(call uv_run,nla) python server.py --port 5009

nla-openapi: ## NLA: rewrite apps/nla/openapi.json from the pydantic models
	cd apps/nla && $(call uv_run,nla) python dump_openapi.py

# ------------------------------------------------------------------- engine --
#
# interp-engine lives in its own repository now (decoderesearch/interp-engine), and inference, graph
# and nla each depend on a pinned release of it from PyPI. Its own tests, lint and docs are there.
#
# These targets are for the one workflow the split made awkward: changing the engine and a service
# that calls it at the same time. `engine-link` swaps an app's venv over to a local engine checkout
# and `engine-unlink` puts the pinned release back.
#
# The state lives in a gitignored marker file, NOT in pyproject.toml or uv.lock, which is the whole
# point: those two are committed, and an accidentally committed local path breaks every other
# checkout and CI with a `Distribution not found at: file:///...`. Nothing tracked changes in either
# direction here, so there is no diff to leak. `make engine-status` says which apps are linked, and
# the -dev targets above print a banner when they start against a linked engine.

# Apps that depend on the engine, and the extras each needs when linked -- an editable install of
# the bare package would drop the extras' own dependency edges from the venv.
ENGINE_APPS = inference graph nla
ENGINE_EXTRAS_inference = [vllm,quant]
ENGINE_EXTRAS_graph =
ENGINE_EXTRAS_nla = [vllm]

# Where your engine checkout lives. Defaults to a sibling of this repo, which is the layout a plain
# `git clone` of both into the same directory produces -- and, being relative, is the same default on
# everyone's machine rather than one person's home directory. Override it per invocation, or export
# it once in your shell. Absolute, or relative to this repo's root; make does not expand `~`, so
# spell out $HOME if you want your home directory:
#   make engine-link APP=inference ENGINE_SRC=$HOME/src/interp-engine
ENGINE_SRC ?= ../interp-engine

# `uv run` for an app, adding --no-sync only when that app is linked. Without it the auto-sync that
# every `uv run` performs would quietly reinstall the pinned wheel over the editable checkout.
uv_run = uv run $(if $(wildcard apps/$(1)/.engine-linked),--no-sync,)

# `uv sync` reinstalls the pinned wheel straight over an editable checkout, so the -install targets
# re-apply the link afterwards rather than silently dropping it. Run from inside apps/<app>, where
# the marker is `.engine-linked`.
engine_relink = $(if $(wildcard apps/$(1)/.engine-linked), && uv pip install -q -e "$$(cat .engine-linked)$(ENGINE_EXTRAS_$(1))",)

# Printed by the -dev targets so a linked venv is visible rather than something you have to remember.
engine_banner = $(if $(wildcard apps/$(1)/.engine-linked),@echo -e "\033[1;33m[engine] $(1) is running the LOCAL engine at $$(cat apps/$(1)/.engine-linked). 'make engine-unlink APP=$(1)' to restore the pinned release.\033[0m",@true)

engine-link: ## Engine: point an app's venv at a local engine checkout. Required: APP=inference
	@test -n "$(APP)" || { echo "Error: APP not set. One of: $(ENGINE_APPS)"; exit 1; }
	@echo "$(ENGINE_APPS)" | tr ' ' '\n' | grep -qx "$(APP)" || { echo "Error: '$(APP)' does not depend on the engine. One of: $(ENGINE_APPS)"; exit 1; }
	@test -f "$(abspath $(ENGINE_SRC))/pyproject.toml" || { echo "Error: no engine checkout at $(abspath $(ENGINE_SRC)). Clone decoderesearch/interp-engine, or set ENGINE_SRC."; exit 1; }
	cd apps/$(APP) && uv sync && uv pip install -e "$(abspath $(ENGINE_SRC))$(ENGINE_EXTRAS_$(APP))"
	@echo "$(abspath $(ENGINE_SRC))" > apps/$(APP)/.engine-linked
	@echo "$(APP) now uses the engine at $(abspath $(ENGINE_SRC)). Undo with: make engine-unlink APP=$(APP)"

engine-unlink: ## Engine: restore an app to its pinned engine release. Required: APP=inference
	@test -n "$(APP)" || { echo "Error: APP not set. One of: $(ENGINE_APPS)"; exit 1; }
	@rm -f apps/$(APP)/.engine-linked
	cd apps/$(APP) && uv sync --exact
	@echo "$(APP) restored to the engine version pinned in its uv.lock."

# Reports what each venv actually resolves, not just what the marker claims. The two can disagree:
# any bare `uv run` or `uv sync` outside these targets reinstalls the pinned wheel over the editable
# checkout, leaving a marker that lies. uv records an editable install in the dist-info's
# direct_url.json, which is a cheap read and needs no import.
engine-status: ## Engine: show which apps are linked to a local engine checkout
	@for app in $(ENGINE_APPS); do \
		marker=""; [ -f apps/$$app/.engine-linked ] && marker=$$(cat apps/$$app/.engine-linked); \
		pin=$$(grep -m1 'interp-engine.*==' apps/$$app/pyproject.toml | sed 's/.*==//; s/".*//'); \
		if [ ! -d apps/$$app/.venv ]; then \
			echo -e "  $$app: \033[2mno venv\033[0m (pins $$pin) -- run 'make $$app-install'"; \
			continue; \
		fi; \
		live=""; \
		url=$$(ls apps/$$app/.venv/lib/python*/site-packages/interp_engine-*.dist-info/direct_url.json 2>/dev/null | head -1); \
		[ -n "$$url" ] && live=$$(sed 's|.*"url":"file://||; s|".*||' "$$url"); \
		if [ -n "$$marker" ] && [ "$$marker" = "$$live" ]; then \
			echo -e "  $$app: \033[33mlocal\033[0m  $$marker"; \
		elif [ -n "$$marker" ]; then \
			echo -e "  $$app: \033[31mstale\033[0m  marker says $$marker, but the venv has $${live:-the pinned wheel} -- 'make engine-link APP=$$app'"; \
		elif [ -n "$$live" ]; then \
			echo -e "  $$app: \033[31mdrift\033[0m  venv has an editable engine at $$live with no marker -- 'make engine-link APP=$$app' or 'make engine-unlink APP=$$app'"; \
		else \
			echo -e "  $$app: \033[32mpinned\033[0m $$pin"; \
		fi; \
	done

# ------------------------------------------------------------------ git hooks --
#
# .githooks/pre-commit runs the fast half of CI -- ruff, eslint, prettier, the spec and config
# scripts -- over the files a commit touches. Opt-in, and per checkout: git stores core.hooksPath
# in .git/config, which is not tracked, so nothing here changes what anyone else's git does.
#
# This target is the Python-side entry point. Webapp contributors get the same thing from the
# `prepare` script in apps/webapp/package.json, which npm runs after every install -- so the two
# halves of the repo each pick the hooks up through the tool they already run.

githooks-install: ## Git hooks: run the fast CI checks before each commit (see .githooks/)
	@git config core.hooksPath .githooks
	@echo "Git hooks enabled for this checkout. Skip one commit with 'git commit --no-verify'."
	@echo "Turn them off with 'make githooks-uninstall'."

githooks-uninstall: ## Git hooks: stop running the pre-commit checks in this checkout
	@git config --unset core.hooksPath 2>/dev/null || true
	@echo "Git hooks disabled for this checkout."

# --------------------------------------------------------------- python lint --
#
# The same gate python-lint.yml runs, for every app in PYTHON_APPS. Each app gets a throwaway
# dev-group-only venv under .lint-venvs/: `uv sync --only-dev` would otherwise strip torch and
# friends out of that app's real .venv, and `--frozen` pins ruff to the app's uv.lock so a local
# run and a CI run cannot disagree. Installing just the dev group takes seconds, not minutes.
#
# LINT_APPS narrows the run to a subset, e.g. `make python-lint LINT_APPS=inference`. The
# pre-commit hook uses it to lint only the apps a commit touches; CI always lints all of them.
LINT_APPS ?= $(PYTHON_APPS)

define run_ruff
	@for app in $(LINT_APPS); do \
		echo -e "\033[1;36m==> $$app\033[0m"; \
		( cd apps/$$app \
			&& UV_PROJECT_ENVIRONMENT=../../.lint-venvs/$$app uv sync --only-dev --frozen -q \
			&& UV_PROJECT_ENVIRONMENT=../../.lint-venvs/$$app uv run --no-sync $(1) ) || exit 1; \
	done
endef

python-lint: ## Python: lint + format checks for every app (what python-lint.yml gates on)
	$(call run_ruff,ruff check .)
	$(call run_ruff,ruff format --check .)
	@python3 .github/scripts/check_lint_config_parity.py
	@python3 .github/scripts/check_no_local_path_deps.py

python-lint-fix: ## Python: apply ruff's autofixes and reformat every app in place
	$(call run_ruff,ruff check --fix .)
	$(call run_ruff,ruff format .)

# Mirrors .github/workflows/agent-rules.yml. Checks the wiring, not the prose: that every
# AGENTS.md is reachable from every harness, and that no rule has been parked somewhere only
# one of them looks.
agent-rules-check: ## Agents: verify AGENTS.md is readable by every coding agent
	@python3 .github/scripts/check_agent_rules.py

# The one command for "I changed a pydantic model". Regenerating a spec and regenerating the
# webapp types that consume it are two separate targets, and the second is the easy one to forget
# -- at which point tsc still compiles against the old shape and the mistake surfaces in CI rather
# than here.
#
# A service that is not installed is skipped rather than fatal, so a webapp-only checkout can
# still refresh its .d.ts from the committed specs. Skipping is reported, because what it means is
# "that spec may now be stale"; the drift test in that app's own suite is what will say so.
#
# Each <app>-openapi target runs dump_openapi.py through uv_run itself rather than delegating to
# apps/<app>/Makefile, whose own `openapi` target is a bare `uv run`. Importing a server is running
# its code, so it needs the same --no-sync an engine-linked app gets from the -dev targets;
# without it, regenerating a spec reinstalls the pinned wheel over the editable checkout and the
# import then fails on whatever the local engine added.
openapi: ## OpenAPI: regenerate every installed service's spec, then the webapp types
	@skipped=""; \
	for app in $(PYTHON_APPS); do \
		if [ -d "apps/$$app/.venv" ]; then \
			echo -e "\033[1;36m==> $$app\033[0m"; \
			$(MAKE) --no-print-directory $$app-openapi || exit 1; \
		else \
			skipped="$$skipped $$app"; \
		fi; \
	done; \
	if [ -d "apps/webapp/node_modules" ]; then \
		echo -e "\033[1;36m==> webapp types\033[0m"; \
		$(MAKE) --no-print-directory webapp-openapi || exit 1; \
	else \
		skipped="$$skipped webapp"; \
	fi; \
	if [ -n "$$skipped" ]; then \
		echo -e "\033[1;33mSkipped, not installed:$$skipped\033[0m"; \
		echo "  Their committed specs are untouched. If you edited models there, run"; \
		echo "  'make <name>-install' first -- CI checks the link either way."; \
	fi

# Mirrors the spec-invariants job in .github/workflows/openapi-drift.yml. Reads the committed
# specs only, so it needs no service installed and answers in well under a second.
openapi-check: ## OpenAPI: verify the committed specs follow the wire-format rules
	@python3 .github/scripts/check_openapi_specs.py
