# Default to nocuda
BUILD_TYPE ?= nocuda

help:  ## Show available commands
	@echo "\n\033[1;35mThe pattern for commands is generally 'make [app]-[environment]-[action]''.\nFor example, 'make webapp-demo-build' will _build_ the _webapp for the demo environment.\033[0m"
	@awk 'BEGIN {FS = ":.*## "; printf "\n"} /^[a-zA-Z_-]+:.*## / { printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init-env: ## Initialize the environment
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

webapp-demo-build: ## Webapp: Public Demo Environment - Build
	@echo "Building the webapp for connecting to the public demo database and servers..."
	@if ! which docker > /dev/null 2>&1; then \
		echo "Error: Docker is not installed. Please install Docker first."; \
		exit 1; \
	fi
	ENV_FILE=.env.demo docker compose build webapp

webapp-demo-run: ## Webapp: Public Demo Environment - Run
	@echo "Bringing up the webapp and connecting to the demo database..."
	@if ! which docker > /dev/null 2>&1; then \
		echo "Error: Docker is not installed. Please install Docker first."; \
		exit 1; \
	fi
	ENV_FILE=.env.demo docker compose --env-file .env.demo --env-file .env up webapp


webapp-demo-check: ## Webapp: Public Demo Environment - Check Config
	@echo "Printing the webapp configuration - this is useful to see if your environment variables are set correctly."
	ENV_FILE=.env.demo docker compose config webapp


webapp-localhost-build: ## Webapp: Localhost Environment - Build (Production Build)
	@echo "Building the webapp for connecting to the localhost database..."
	@if ! which docker > /dev/null 2>&1; then \
		echo "Error: Docker is not installed. Please install Docker first."; \
		exit 1; \
	fi
	ENV_FILE=.env.localhost docker compose build webapp db-init postgres

webapp-localhost-run: ## Webapp: Localhost Environment - Run (Production Build)
	@echo "Bringing up the webapp and connecting to the localhost database..."
	@if ! which docker > /dev/null 2>&1; then \
		echo "Error: Docker is not installed. Please install Docker first."; \
		exit 1; \
	fi
	ENV_FILE=.env.localhost docker compose --env-file .env.localhost --env-file .env up webapp db-init postgres

install-nodejs: # Install Node.js for Webapp
	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
	# Need to source NVM in the same shell
	. ${HOME}/.nvm/nvm.sh && nvm install 22

webapp-localhost-install: ## Webapp: Localhost Environment - Install Dependencies (Development Build)
	@echo "Installing the webapp dependencies for development in the localhost environment..."
	# check if npm exists
	if ! which npm > /dev/null 2>&1; then \
		echo "Error: npm is not installed. Please install nodejs first with 'make install-nodejs'."; \
		exit 1; \
	fi
	cd apps/webapp && \
	npm install
	
webapp-localhost-dev: ## Webapp: Localhost Environment - Run (Development Build)
	@echo "Bringing up the webapp for development and connecting to the localhost database..."
	@if ! which docker > /dev/null 2>&1; then \
		echo "Error: Docker is not installed. Please install Docker first."; \
		exit 1; \
	fi
	ENV_FILE=.env.localhost docker compose \
		-f docker-compose.yaml \
		-f docker-compose.webapp.dev.yaml \
		--env-file .env.localhost \
		--env-file .env \
		up webapp db-init postgres

webapp-localhost-test: ## Webapp: Localhost Environment - Run (Playwright)
	@echo "Bringing up the webapp for development and connecting to the localhost database..."
	@if ! which docker > /dev/null 2>&1; then \
		echo "Error: Docker is not installed. Please install Docker first."; \
		exit 1; \
	fi
	ENV_FILE=.env.localhost docker compose \
		-f docker-compose.yaml \
		-f docker-compose.webapp.test.yaml \
		--env-file .env.localhost \
		--env-file .env \
		up webapp db-init postgres
	

inference-localhost-install: ## Inference: Localhost Environment - Install Dependencies (Development Build)
	@echo "Installing the inference dependencies for development in the localhost environment..."
	cd apps/inference && \
	poetry remove neuronpedia-inference-client || true && \
	poetry add ../../packages/python/neuronpedia-inference-client && \
	poetry lock && poetry install

inference-localhost-build: ## Inference: Localhost Environment - Build
	@echo "Building the inference server for the localhost environment..."
	ENV_FILE=.env.localhost \
		BUILD_TYPE=$(BUILD_TYPE) \
		docker compose \
		-f docker-compose.yaml \
		$(if $(USE_LOCAL_HF_CACHE),-f docker-compose.hf-cache.yaml,) \
		build inference

inference-localhost-build-gpu: ## Inference: Localhost Environment - Build (CUDA). Usage: make inference-localhost-build-gpu [USE_LOCAL_HF_CACHE=1]
	$(MAKE) inference-localhost-build BUILD_TYPE=cuda

inference-localhost-dev: ## Inference: Localhost Environment - Run (Development Build). Usage: make inference-localhost-dev [MODEL_SOURCESET=gpt2-small.res-jb] [AUTORELOAD=1]
	@echo "Bringing up the inference server for development in the localhost environment..."
	@if [ "$(MODEL_SOURCESET)" != "" ]; then \
		if [ ! -f ".env.inference.$(MODEL_SOURCESET)" ]; then \
			echo "Error: Configuration file .env.inference.$(MODEL_SOURCESET) not found."; \
			echo "Please run 'make inference-list-configs' to see available configurations."; \
			exit 1; \
		fi; \
		echo "Using model configuration: .env.inference.$(MODEL_SOURCESET)"; \
		RELOAD=$$([ "$(AUTORELOAD)" = "1" ] && echo "1" || echo "0") \
		ENV_FILE=.env.inference.$(MODEL_SOURCESET) \
			docker compose \
			-f docker-compose.yaml \
			-f docker-compose.inference.dev.yaml \
			$(if $(ENABLE_GPU),-f docker-compose.inference.gpu.yaml,) \
			$(if $(USE_LOCAL_HF_CACHE),-f docker-compose.hf-cache.yaml,) \
			--env-file .env.inference.$(MODEL_SOURCESET) \
			--env-file .env.localhost \
			--env-file .env \
			up inference; \
	else \
		echo "Error: MODEL_SOURCESET not specified. Please specify a model+source configuration, e.g. to load .env.inference.gpt2-small.res-jb, run: make inference-localhost-dev MODEL_SOURCESET=gpt2-small.res-jb"; \
		echo "Please run 'make inference-list-configs' to see available configurations."; \
		exit 1; \
	fi

inference-localhost-dev-gpu: ## Inference: Localhost Environment - Run (Development Build with CUDA). Usage: make inference-localhost-dev-gpu [MODEL_SOURCESET=gpt2-small.res-jb] [AUTORELOAD=1] [USE_LOCAL_HF_CACHE=1]
	$(MAKE) inference-localhost-dev ENABLE_GPU=1 MODEL_SOURCESET=$(MODEL_SOURCESET) AUTORELOAD=$(AUTORELOAD)

inference-list-configs: ## Inference: List Configurations (possible values for MODEL_SOURCESET)
	@echo "\nAvailable Inference Configurations (.env.inference.*)\n================================================\n"
	@for config in $$(ls .env.inference.*); do \
		name=$$(echo $$config | sed 's/^.env.inference.//'); \
		echo "\033[1;36m$$name\033[0m"; \
		model_id=$$(grep "^MODEL_ID=" $$config | cut -d'=' -f2); \
		sae_sets=$$(grep "^SAE_SETS=" $$config | cut -d'=' -f2); \
		echo "    Model: \033[33m$$model_id\033[0m"; \
		echo "    Source/SAE Sets: \033[32m$$sae_sets\033[0m"; \
		echo "    \033[1;35mmake inference-localhost-dev MODEL_SOURCESET=$$name\033[0m"; \
		echo "    \033[1;35mmake inference-localhost-dev-gpu MODEL_SOURCESET=$$name\033[0m"; \
		echo ""; \
	done

reset-docker-data: ## Reset Docker Data - this deletes your local database!
	@echo "WARNING: This will delete all your local neuronpedia Docker data and databases!"
	@read -p "Are you sure you want to continue? (y/N) " confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi
	@echo "Resetting Docker data..."
	docker compose down -v
