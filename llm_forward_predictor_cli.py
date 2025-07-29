#!/usr/bin/env python3

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np

from batch_benchmark_runner import SimpleBenchmarkRunner
from train_utils import build_stage_features, train_linear_predictor, preprocess_input_for_prediction


def profile_model(
        model_name: str, tp_size: int = 1, pp_size: int = 1, max_batch_size: int = 64, 
        max_input_tokens: int = 16384, output_len: int = 32, num_runs: int = 3, backend: str = "sglang",
        overwrite_cache: bool = False) -> str:
    """Profile a model and save benchmark data."""
    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Import and create backend
    from bench_backend_handler import SGLangBackend, VLLMBackend
    
    if backend == "sglang":
        backend_instance = SGLangBackend()
    elif backend == "vllm":
        backend_instance = VLLMBackend()
    else:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends: sglang, vllm")
    
    runner = SimpleBenchmarkRunner(backend_instance)
    
    results = runner.run_sweep(
        model_path=model_name,
        max_batch_size=max_batch_size,
        max_input_tokens=max_input_tokens,
        output_len=output_len,
        server_args={"load_format": "dummy", "tp_size": tp_size, "pp_size": pp_size},
        use_cache=not overwrite_cache,
        skip_cache=overwrite_cache,
        cache_tag="v1",
        num_runs=num_runs,
    )
    
    # Get GPU model name for filename
    gpu_model = results.get('metadata', {}).get('gpu_model')
    if gpu_model is None:
        # Fallback: detect current GPU if not in metadata
        from batch_benchmark_runner import _get_gpu_info
        gpu_info = _get_gpu_info()
        gpu_model = gpu_info.get('gpu_model', 'unknown')
    
    out_name = f"benchmark_data_{model_name.replace('/','_')}_TP_{tp_size}_PP_{pp_size}_{gpu_model}_{backend}.json"
    with open(out_name, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Profiling complete. Results saved to {out_name}")
    print(f"Successful configs: {len(results.get('results', []))}")
    print(f"Failed configs: {len(results.get('metadata', {}).get('failed_configs', []))}")
    
    return out_name


def train_models(config_name: str, benchmark_file: str, predictor_file: str = "trained_predictors.json") -> None:
    """Train regression models from benchmark data."""
    if not os.path.exists(benchmark_file):
        print(f"Error: Benchmark file {benchmark_file} not found")
        sys.exit(1)
    
    with open(benchmark_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    if not results:
        print("Error: No benchmark results found in benchmark file")
        sys.exit(1)
    
    df = pd.DataFrame(results)
    
    # Load or create predictor storage
    if os.path.exists(predictor_file):
        with open(predictor_file, 'r') as f:
            predictors = json.load(f)
    else:
        predictors = {}
    
    if config_name not in predictors:
        predictors[config_name] = {}
    
    # Add metadata to the config
    predictors[config_name]["metadata"] = metadata
    
    print(f"Training models for config: {config_name}")
    print(f"Training data: {len(df)} benchmark results")
    
    for stage in ['prefill', 'decode']:
        print(f"\nTraining {stage} predictor...")
        
        stage_df = build_stage_features(df, stage)
        stage_df = stage_df.dropna()
        
        if len(stage_df) == 0:
            print(f"Warning: No valid {stage} data found")
            continue
        
        # Train linear model
        model = train_linear_predictor(stage_df, f"{config_name}_{stage}")
        
        # Calculate MSE for the model
        X_train = stage_df[['num_new_tokens', 'prod_ext_ctx', 'num_context_tokens', 'batch_size']].to_numpy(dtype=np.float32)
        y_train = stage_df['time'].to_numpy(dtype=np.float32)
        y_pred = model.predict(X_train)
        mse = np.mean((y_train - y_pred) ** 2)
        
        # Store model parameters including MSE
        predictors[config_name][stage] = {
            "weights": model.coef_.tolist(),
            "bias": float(model.intercept_),
            "model_type": "linear",
            "mse": float(mse)
        }
    
    # Save trained predictors
    with open(predictor_file, 'w') as f:
        json.dump(predictors, f, indent=2)
    
    print(f"\nModels trained and saved to {predictor_file}")


def predict_latency(predictor_file: str, config_name: str, mode: str, batch_size: int, input_len: int) -> None:
    """Make latency predictions using trained models."""
    if not os.path.exists(predictor_file):
        print(f"Error: Predictor file {predictor_file} not found")
        sys.exit(1)
    
    with open(predictor_file, 'r') as f:
        predictors = json.load(f)
    
    if config_name not in predictors:
        print(f"Error: Config '{config_name}' not found in predictor file")
        print(f"Available configs: {list(predictors.keys())}")
        sys.exit(1)
    
    if mode not in predictors[config_name]:
        print(f"Error: Mode '{mode}' not found for config '{config_name}'")
        print(f"Available modes: {list(predictors[config_name].keys())}")
        sys.exit(1)
    
    model_data = predictors[config_name][mode]
    weights = np.array(model_data["weights"])
    bias = model_data["bias"]
    
    # Prepare features
    features = preprocess_input_for_prediction(batch_size, input_len, True, mode)
    
    # Make prediction
    prediction = np.dot(weights, features) + bias
    
    print(f"Prediction for {config_name} ({mode}):")
    print(f"  Batch size: {batch_size}")
    print(f"  Input length: {input_len}")
    print(f"  Predicted latency: {prediction*1000:.2f}ms")


def view_predictions(predictor_file: str = "trained_predictors.json") -> None:
    """Simple viewer for prediction models."""
    if not os.path.exists(predictor_file):
        print(f"No trained predictors found at {predictor_file}. Run 'train_models' first.")
        sys.exit(1)
    
    with open(predictor_file, 'r') as f:
        predictors = json.load(f)
    
    print("Available prediction models:")
    print("=" * 50)
    
    for config_name, config_data in predictors.items():
        print(f"\nConfig: {config_name}")
        for mode, model_data in config_data.items():
            weights = model_data.get("weights", [])
            bias = model_data.get("bias", 0)
            print(f"  {mode}:")
            print(f"    Weights: {[f'{w:.4f}' for w in weights]}")
            print(f"    Bias: {bias:.4f}")
    
    print("\nFeature order: [num_new_tokens, prod_ext_ctx, num_context_tokens, batch_size]")
    
    # Interactive prediction
    while True:
        try:
            print("\nEnter prediction parameters (or 'quit' to exit):")
            config = input("Config name: ").strip()
            if config.lower() == 'quit':
                break
            
            if config not in predictors:
                print(f"Config '{config}' not found")
                continue
            
            mode = input("Mode (prefill/decode): ").strip()
            if mode not in predictors[config]:
                print(f"Mode '{mode}' not found for config '{config}'")
                continue
            
            batch_size = int(input("Batch size: "))
            input_len = int(input("Input length: "))
            
            predict_latency(predictor_file, config, mode, batch_size, input_len)
            
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            break


def launch_web_viewer(predictor_file: str = "trained_predictors.json", host: str = "0.0.0.0", port: int = 7860) -> None:
    """Launch the web-based viewer using Gradio."""
    try:
        from llm_predictor_viewer import demo
        print(f"Launching web viewer at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        demo.launch(share=False, server_name=host, server_port=port)
    except ImportError as e:
        print(f"Error importing web viewer: {e}")
        print("Make sure all required dependencies are installed (gradio, plotly)")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching web viewer: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="LLM Forward Time Profiler CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile a model')
    profile_parser.add_argument('model_name', help='Model name or path')
    profile_parser.add_argument('--tp_size', type=int, default=1, help='Tensor parallel size')
    profile_parser.add_argument('--pp_size', type=int, default=1, help='Pipeline parallel size')
    profile_parser.add_argument('--max_batch_size', type=int, default=64, help='Maximum batch size to test')
    profile_parser.add_argument('--max_input_tokens', type=int, default=16384, help='Maximum input tokens to test')
    profile_parser.add_argument('--output_len', type=int, default=32, help='Output length for decode phase')
    profile_parser.add_argument('--num_runs', type=int, default=3, help='Number of runs per configuration')
    profile_parser.add_argument('--backend', default='sglang', choices=['sglang', 'vllm'], help='Backend to use for profiling (default: sglang)')
    profile_parser.add_argument('--overwrite-cache', action='store_true', help='Force overwrite of existing cache data.')
    
    # Train models command
    train_parser = subparsers.add_parser('train_models', help='Train regression models from benchmark data')
    train_parser.add_argument('config_name', help='Name for this configuration')
    train_parser.add_argument('benchmark_file', help='Path to benchmark data JSON file')
    train_parser.add_argument('--predictor-file', default='trained_predictors.json', help='Path to output predictor file (default: trained_predictors.json)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make latency predictions')
    predict_parser.add_argument('predictor_file', help='Path to trained predictors JSON file')
    predict_parser.add_argument('config_name', help='Configuration name to use')
    predict_parser.add_argument('--mode', choices=['prefill', 'decode'], required=True, help='Prediction mode')
    predict_parser.add_argument('--bs', type=int, required=True, help='Batch size')
    predict_parser.add_argument('--input-len', dest='input_len', type=int, required=True, help='Input length')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View trained models and make interactive predictions')
    view_parser.add_argument('--predictor-file', default='trained_predictors.json', help='Path to predictor file (default: trained_predictors.json)')
    
    # Web viewer command
    webview_parser = subparsers.add_parser('webview', help='Launch web-based interactive viewer with plots')
    webview_parser.add_argument('--predictor-file', default='trained_predictors.json', help='Path to predictor file (default: trained_predictors.json)')
    webview_parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server (default: 0.0.0.0)')
    webview_parser.add_argument('--port', type=int, default=7860, help='Port to bind the server (default: 7860)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'profile':
        profile_model(
            args.model_name,
            args.tp_size,
            args.pp_size,
            args.max_batch_size,
            args.max_input_tokens,
            args.output_len,
            args.num_runs,
            args.backend,
            args.overwrite_cache
        )
    elif args.command == 'train_models':
        train_models(args.config_name, args.benchmark_file, args.predictor_file)
    elif args.command == 'predict':
        predict_latency(args.predictor_file, args.config_name, args.mode, args.bs, args.input_len)
    elif args.command == 'view':
        view_predictions(args.predictor_file)
    elif args.command == 'webview':
        launch_web_viewer(args.predictor_file, args.host, args.port)


if __name__ == "__main__":
    main()