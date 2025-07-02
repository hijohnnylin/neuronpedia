# Makefile for OpenAPI client generation
# Variables
INFERENCE_SPEC = openapi/inference-server.yaml
AUTOINTERP_SPEC = openapi/autointerp-server.yaml
TS_OUTPUT_DIR = ../packages/typescript
PY_OUTPUT_DIR = ../packages/python
IGNORE_FILE = .openapi-generator-ignore

.PHONY: generate-inference-clients generate-autointerp-clients clean-inference-clients clean-autointerp-clients \
        inference-setup-env autointerp-setup-env webapp-setup-env-inference webapp-setup-env-autointerp

# Check if VERSION is provided
check-version:
ifndef VERSION
	$(error VERSION is required. Please specify with VERSION=x.y.z)
endif

# Clean and generate inference clients
generate-inference-clients: check-version clean-inference-clients
	@echo "Generating Python inference client with version $(VERSION)..."
	openapi-generator generate \
		-i $(INFERENCE_SPEC) \
		-g python \
		-o $(PY_OUTPUT_DIR)/neuronpedia-inference-client \
		--package-name neuronpedia_inference_client \
		--additional-properties=packageVersion=$(VERSION) \
		--ignore-file-override=$(IGNORE_FILE)

	@echo "Generating TypeScript inference client..."
	openapi-generator generate \
		-i $(INFERENCE_SPEC) \
		-g typescript-fetch \
		-o $(TS_OUTPUT_DIR)/neuronpedia-inference-client \
		-p npmName=neuronpedia-inference-client,npmVersion=$(VERSION) \
		--ignore-file-override=$(IGNORE_FILE)

# Clean and generate autointerp clients
generate-autointerp-clients: check-version clean-autointerp-clients
	@echo "Generating Python autointerp client with version $(VERSION)..."
	openapi-generator generate \
		-i $(AUTOINTERP_SPEC) \
		-g python \
		-o $(PY_OUTPUT_DIR)/neuronpedia-autointerp-client \
		--package-name neuronpedia_autointerp_client \
		--additional-properties=packageVersion=$(VERSION) \
		--ignore-file-override=$(IGNORE_FILE)

	@echo "Generating TypeScript autointerp client..."
	openapi-generator generate \
		-i $(AUTOINTERP_SPEC) \
		-g typescript-fetch \
		-o $(TS_OUTPUT_DIR)/neuronpedia-autointerp-client \
		-p npmName=neuronpedia-autointerp-client,npmVersion=$(VERSION) \
		--ignore-file-override=$(IGNORE_FILE)

# Setup inference server environment
inference-setup-env:
	@echo "Setting up inference server environment..."
	@cd ../apps/inference && \
	poetry remove neuronpedia-inference-client && \
	poetry add ../../packages/python/neuronpedia-inference-client/

# Setup autointerp server environment
autointerp-setup-env:
	@echo "Setting up autointerp server environment..."
	@cd ../apps/autointerp && \
	poetry add ../../packages/python/neuronpedia-autointerp-client

# Setup webapp environment for inference
webapp-setup-env-inference:
	@echo "Setting up webapp environment for inference client..."
	@if ! command -v typescript &> /dev/null; then \
		echo "Installing TypeScript globally..."; \
		npm install -g typescript; \
	fi
	@cd ../apps/webapp && \
	npm link ../../packages/typescript/neuronpedia-inference-client && \
	npm link neuronpedia-inference-client

# Setup webapp environment for autointerp
webapp-setup-env-autointerp:
	@echo "Setting up webapp environment for autointerp client..."
	@if ! command -v typescript &> /dev/null; then \
		echo "Installing TypeScript globally..."; \
		npm install -g typescript; \
	fi
	@cd ../apps/webapp && \
	npm link ../../packages/typescript/neuronpedia-autointerp-client && \
	npm link neuronpedia-autointerp-client

# Clean up inference clients
clean-inference-clients:
	@echo "Cleaning inference clients..."
	rm -rf $(PY_OUTPUT_DIR)/neuronpedia-inference-client
	rm -rf $(TS_OUTPUT_DIR)/neuronpedia-inference-client

# Clean up autointerp clients
clean-autointerp-clients:
	@echo "Cleaning autointerp clients..."
	rm -rf $(PY_OUTPUT_DIR)/neuronpedia-autointerp-client
	rm -rf $(TS_OUTPUT_DIR)/neuronpedia-autointerp-client

# Setup all for inference
setup-all-inference: generate-inference-clients inference-setup-env webapp-setup-env-inference
	@echo "All inference client environments set up successfully!"

# Setup all for autointerp
setup-all-autointerp: generate-autointerp-clients autointerp-setup-env webapp-setup-env-autointerp
	@echo "All autointerp client environments set up successfully!"

# Help
help:
	@echo "OpenAPI Client Generator Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  generate-inference-clients   - Generate Python and TypeScript clients for inference server"
	@echo "  generate-autointerp-clients  - Generate Python and TypeScript clients for autointerp server"
	@echo "  inference-setup-env          - Setup inference server environment with the new clients"
	@echo "  autointerp-setup-env         - Setup autointerp server environment with the new clients"
	@echo "  webapp-setup-env-inference   - Setup webapp environment with the new inference clients" 
	@echo "  webapp-setup-env-autointerp  - Setup webapp environment with the new autointerp clients"
	@echo "  setup-all-inference          - Do all setup steps for inference clients"
	@echo "  setup-all-autointerp         - Do all setup steps for autointerp clients"
	@echo "  clean-inference-clients      - Remove existing inference clients"
	@echo "  clean-autointerp-clients     - Remove existing autointerp clients"
	@echo ""
	@echo "Required options:"
	@echo "  VERSION=x.y.z                - Set the semantic version for both Python and TypeScript clients"
	@echo ""
	@echo "Example usage:"
	@echo "  make generate-inference-clients VERSION=1.2.3"
	@echo "  make setup-all-inference VERSION=1.2.3"
	@echo "  make setup-all-autointerp VERSION=1.2.3"
