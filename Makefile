.PHONY: clean build test-upload upload install dev-install test test-minimal help

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

test: test-minimal

test-minimal: 
	@echo "Running ultra-minimal end-to-end test..."
	@mkdir -p test_output
	@echo "1. CLI smoke test..."
	@python -m llm_execution_time_predictor.llm_forward_predictor_cli --help > /dev/null && echo "✓ CLI works" || (echo "✗ CLI failed" && exit 1)
	
	@echo "2. Test prefill profiling..."
	@python -m llm_execution_time_predictor.llm_forward_predictor_cli profile prefill \
		--model-path Qwen/Qwen3-4B \
		--load-format dummy \
		--batch-size 1 \
		--input-len 64 \
		--max-batch-size 1 \
		--max-input-len 64 \
		--output-dir test_output
	@test -s test_output/prefill_prefill_profiling_tp0.jsonl && echo "✓ Prefill output generated" || (echo "✗ Prefill failed" && exit 1)
	@head -1 test_output/prefill_prefill_profiling_tp0.jsonl | python -c "import json,sys; json.load(sys.stdin); print('✓ Prefill JSON valid')" 2>/dev/null || (echo "✗ Prefill JSON invalid" && exit 1)
	
	@echo "3. Test prefill with cache profiling..."
	@python -m llm_execution_time_predictor.llm_forward_predictor_cli profile prefill-prefix-cache \
		--model-path Qwen/Qwen3-4B \
		--load-format dummy \
		--batch-size 1 \
		--input-len 64 \
		--max-batch-size 1 \
		--max-input-len 64 \
		--output-dir test_output
	@test -s test_output/prefill_profiling_chunked_cache_prefix_caching_prefill_cache_profiling_tp0.jsonl && echo "✓ Prefill-cache output generated" || (echo "✗ Prefill-cache failed" && exit 1)
	@head -1 test_output/prefill_profiling_chunked_cache_prefix_caching_prefill_cache_profiling_tp0.jsonl | python -c "import json,sys; json.load(sys.stdin); print('✓ Prefill-cache JSON valid')" 2>/dev/null || (echo "✗ Prefill-cache JSON invalid" && exit 1)
	
	@echo "4. Test decode profiling..."
	@python -m llm_execution_time_predictor.llm_forward_predictor_cli profile decode \
		--model-path Qwen/Qwen3-4B \
		--load-format dummy \
		--batch-size 1 \
		--input-len 64 \
		--max-batch-size 1 \
		--max-input-len 64 \
		--output-dir test_output
	@test -s test_output/decode_decode_profiling_tp0.jsonl && echo "✓ Decode output generated" || (echo "✗ Decode failed" && exit 1)
	@head -1 test_output/decode_decode_profiling_tp0.jsonl | python -c "import json,sys; json.load(sys.stdin); print('✓ Decode JSON valid')" 2>/dev/null || (echo "✗ Decode JSON invalid" && exit 1)
	
	@echo "5. Test real workload profiling..."
	@python -m llm_execution_time_predictor.llm_forward_predictor_cli profile_real \
		--model Qwen/Qwen3-4B \
		--output_file test_output/test_real.jsonl \
		--max_job_send_time 1 \
		--max_rps 2 \
		--max_window_time 10 \
		--data_file llm_execution_time_predictor/monkey_patch_sglang/data/splitwise_code.csv
	@test -s test_output/test_real.jsonl && echo "✓ Real workload output generated" || (echo "✗ Real workload failed" && exit 1)
	@head -1 test_output/test_real.jsonl | python -c "import json,sys; json.load(sys.stdin); print('✓ Real workload JSON valid')" 2>/dev/null || (echo "✗ Real workload JSON invalid" && exit 1)
	
	@echo "6. Cleanup..."
	@rm -rf test_output/
	@echo "✓ All 4 modes tested successfully!"

# Full workflow targets
test-release: clean build test-install test-upload ## Full test release workflow
	@echo "Test release complete! Check https://test.pypi.org/project/$(PACKAGE_NAME)/"

release: clean build test-install check upload ## Full production release workflow
	@echo "Production release complete! Check https://pypi.org/project/$(PACKAGE_NAME)/"

test-profile:
	$(MAKE) test-profile-model MODEL_PATH=Qwen/Qwen3-8B TP=1

test-profile-model:
	@if [ -z "$(MODEL_PATH)" ]; then echo "MODEL_PATH is required. Usage: make test-profile-model MODEL_PATH=path/to/model"; exit 1; fi
	@echo "Testing profiling with $(MODEL_PATH)..."
	@mkdir -p profile_output
	@echo "1. Profile prefill..."
	python3 -m llm_execution_time_predictor.llm_forward_predictor_cli profile prefill --model-path $(MODEL_PATH) --load-format dummy --output-dir profile_output --tp-size $(TP)
# 	@echo "2. Profile prefill with cache..."
# 	python3 -m llm_execution_time_predictor.llm_forward_predictor_cli profile prefill-prefix-cache --model-path $(MODEL_PATH) --load-format dummy --output-dir profile_output --tp-size $(TP)
# 	@echo "3. Profile decode..."
# 	python3 -m llm_execution_time_predictor.llm_forward_predictor_cli profile decode --model-path $(MODEL_PATH) --load-format dummy --output-dir profile_output --tp-size $(TP)
# 	@echo "4. Profile real workload Splitwise Code RPS=5..."
# 	python3 -m llm_execution_time_predictor.llm_forward_predictor_cli profile_real --model $(MODEL_PATH) --output_file profile_output/splitwise_code_rps_5.jsonl --max_job_send_time 60 --max_rps 5 --max_window_time 300 --data_file llm_execution_time_predictor/monkey_patch_sglang/data/splitwise_code.csv --tp $(TP)
# 	@echo "4. Profile real workload Splitwise Code RPS=10..."
# 	python3 -m llm_execution_time_predictor.llm_forward_predictor_cli profile_real --model $(MODEL_PATH) --output_file profile_output/splitwise_code_rps_10.jsonl --max_job_send_time 60 --max_rps 2 --max_window_time 300 --data_file llm_execution_time_predictor/monkey_patch_sglang/data/splitwise_code.csv --tp $(TP)
# 	@echo "4. Profile real workload Arxiv Summarization RPS=3..."
# 	python3 -m llm_execution_time_predictor.llm_forward_predictor_cli profile_real --model $(MODEL_PATH) --output_file profile_output/arxiv_summarization_rps_3.jsonl --max_job_send_time 60 --max_rps 3 --max_window_time 300 --data_file llm_execution_time_predictor/monkey_patch_sglang/data/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv --tp $(TP)
# 	@echo "Profile testing completed!"