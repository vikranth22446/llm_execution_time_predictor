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