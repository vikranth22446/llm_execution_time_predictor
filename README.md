# LLM Execution Time Predictor

A small utility to help train a regression model given to predict prefill/decode times. 
By using the batch size and input, the prefill/decode execution times are very predictable.

This can be plugged into a simulator for faster experiments.

A more complicated version is done by https://github.com/microsoft/vidur but it trains every component of the model forwarding. This utility instead just profiles the full model forwarding as a unit to simplify research.

The tool https://modal.com/llm-almanac/advisor is nice visualizer but it doesn't let you train a local version and specify an exact bs/input

## Installation

### Install from Source
```bash
pip install -r requirements.txt
```

## Using Prefill/Decode execution time for predictors
A very small set of features are used to train the predictor.
Num new tokens: total tokens processed/generated:
- for decode, it's the batch size. for prefill, it's the full input chunk
Product ext cost: Represents the cost of attention
- For prefill, it's O(seq_len^2) so we do bs * input^2
- For decode, it's just O(seq_len)
Total context tokens: 
- Total tokens processed across batch * input representing the cache usage
Time of kernel

Tested on both prefill/decode the decode time 

## Usage

### Using the PyPI Package
```bash
# Profile prefill performance
llm-execution-time-predictor profile prefill --model-path <model_name>

# Profile decode performance
llm-execution-time-predictor profile decode --model-path <model_name> --max-decode-token-length <length>

# Profile prefill performance with prefix cache
llm-execution-time-predictor profile prefill-prefix-cache --model-path <model_name>

# Profile real workload using monkey patching
llm-execution-time-predictor profile_real --model <model_name> --output_file <output.jsonl>
```

### Examples
```bash
# Profile prefill with dummy weights for testing
python3 -m llm_execution_time_predictor.llm_forward_predictor_cli profile prefill --model-path Qwen/Qwen3-4B --load-format dummy

# Profile decode with specific token length limit
python -m llm_execution_time_predictor.llm_forward_predictor_cli profile decode --model-path Qwen/Qwen3-4B --max-decode-token-length 512

# Profile prefill with prefix cache
python -m llm_execution_time_predictor.llm_forward_predictor_cli profile prefill-prefix-cache --model-path Qwen/Qwen3-4B --load-format dummy

# Profile real workload with custom parameters
python -m llm_execution_time_predictor.llm_forward_predictor_cli profile_real --model Qwen/Qwen3-4B --output_file profile_results.jsonl --max_job_send_time 10
```

The trained predictor file format:
```json
{
    "config_name": {
        "prefill": {
            "weights": [0.1234, 0.5678, 0.9012, 0.3456],
            "bias": 0.0123,
            "model_type": "linear"
        },
        "decode": {
            "weights": [0.2345, 0.6789, 0.0123, 0.4567],
            "bias": 0.0456,
            "model_type": "linear"
        }
    }
}
```

Feature order: `[num_new_tokens, prod_ext_ctx, num_context_tokens, batch_size]`

# Downloading/Saving Model Files
In order to track files, the repo uses [gdvc](https://github.com/vikranth22446/gdvc_mini)

To download latest public files:
```
gdvc update
```
Check the gdvc docs for adding more or using your root.

# Ack
Co-contributors: [Dongming Li](https://github.com/dongmingli-Ben) and [Zijian He](https://github.com/jiange91) helped inspired the workflow
