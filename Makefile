.PHONY: clean build test-upload upload install dev-install help

# Variables
PACKAGE_NAME = llm_execution_time_predictor
DIST_DIR = dist/
BUILD_DIR = build/

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts
	rm -rf $(DIST_DIR) $(BUILD_DIR) *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

install-tools: ## Install build and upload tools
	pip install --upgrade build twine

build: clean install-tools ## Build the package
	python -m build

test-install: build ## Test install the built package locally
	pip uninstall $(PACKAGE_NAME) -y || true
	pip install $(DIST_DIR)*.whl

test-upload: build ## Upload to TestPyPI (requires TestPyPI account and API token)
	@echo "Uploading to TestPyPI..."
	@echo "Make sure you have set up API tokens in ~/.pypirc or environment variables"
	twine upload --repository testpypi $(DIST_DIR)*

upload: build ## Upload to production PyPI (requires PyPI account and API token)
	@echo "Uploading to production PyPI..."
	@echo "Make sure you have set up API tokens in ~/.pypirc or environment variables"
	@echo "Make sure you've tested with 'make test-upload' first!"
	@read -p "Are you sure you want to upload to production PyPI? (y/N): " confirm && [ "$$confirm" = "y" ]
	twine upload $(DIST_DIR)*

dev-install: ## Install package in development mode
	pip install -e .

check: ## Check package metadata and description
	twine check $(DIST_DIR)*

# Full workflow targets
test-release: clean build test-install test-upload ## Full test release workflow
	@echo "Test release complete! Check https://test.pypi.org/project/$(PACKAGE_NAME)/"

release: clean build test-install check upload ## Full production release workflow
	@echo "Production release complete! Check https://pypi.org/project/$(PACKAGE_NAME)/"

test-profile:
	$(MAKE) test-profile-model MODEL_PATH=Qwen/Qwen3-4B

test-profile-model:
	@if [ -z "$(MODEL_PATH)" ]; then echo "MODEL_PATH is required. Usage: make test-profile-model MODEL_PATH=path/to/model"; exit 1; fi
	@echo "Testing profiling with $(MODEL_PATH)..."
# 	@echo "1. Profile prefill..."
# 	python llm_execution_time_predictor/llm_forward_predictor_cli.py profile prefill --model-path $(MODEL_PATH) --load-format dummy
# 	@echo "2. Profile decode..."
# 	python llm_execution_time_predictor/llm_forward_predictor_cli.py profile decode --model-path $(MODEL_PATH) --max-decode-token-length 512 --load-format dummy
	@echo "3. Profile real workload..."
	python llm_execution_time_predictor/llm_forward_predictor_cli.py profile_real --model $(MODEL_PATH) --output_file test_profile_results.jsonl --max_job_send_time 5 --max_rps 5 --data_file llm_execution_time_predictor/monkey_patch_sglang/data/splitwise_code.csv
	@echo "Profile testing completed!"