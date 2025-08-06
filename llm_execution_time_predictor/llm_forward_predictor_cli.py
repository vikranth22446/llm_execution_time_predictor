import argparse
import os
import subprocess
import sys
from pathlib import Path

# Import the existing argument parsing functions
try:
    from sglang_batch_latency import ServerArgs, BenchArgs, main as sglang_main
except ImportError as e:
    from .sglang_batch_latency import ServerArgs, BenchArgs, main as sglang_main
    
# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
MONKEY_PATCH_SCRIPT = SCRIPT_DIR / "monkey_patch_sglang" / "run_splitwise_file.py"

def profile_prefill(server_args, bench_args):
    """Profile prefill performance using SGLang batch latency."""
    bench_args.run_prefill_profiling = True
    bench_args.run_decode_profiling = False
    bench_args.correctness_test = False
    sglang_main(server_args, bench_args)

def profile_decode(server_args, bench_args):
    """Profile decode performance using SGLang batch latency."""
    bench_args.run_prefill_profiling = False
    bench_args.run_decode_profiling = True
    bench_args.correctness_test = False
    sglang_main(server_args, bench_args)

def profile_real(args):
    """Profile real workload using monkey patching script."""
    env = os.environ.copy()
    
    # Set the profiling output environment variable
    if args.output_file:
        env['SGLANG_PROFILE_OUTPUT'] = args.output_file
    
    cmd = [
        sys.executable, str(MONKEY_PATCH_SCRIPT),
        "--model", args.model,
        "--max_job_send_time", str(args.max_job_send_time)
    ]
    if args.data_file:
        cmd.extend(["--data_file", args.data_file])
    
    if args.max_rps:
        cmd.extend(["--max_rps", str(args.max_rps)])
    
    if args.rps_scale:
        cmd.extend(["--rps_scale", str(args.rps_scale)])
    
    if args.output_file:
        cmd.extend(["--output_file", args.output_file])
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Environment: SGLANG_PROFILE_OUTPUT={env.get('SGLANG_PROFILE_OUTPUT', 'Not set')}")
    return subprocess.run(cmd, env=env)

def main():
    parser = argparse.ArgumentParser(
        description="LLM Forward Predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Profile prefill performance
            %(prog)s profile prefill --model-path meta-llama/Meta-Llama-3-8B-Instruct

            # Profile decode performance  
            %(prog)s profile decode --model-path meta-llama/Meta-Llama-3-8B-Instruct --max-decode-token-length 8192

            # Profile real workload
            %(prog)s profile_real --model Qwen/Qwen3-8B --output_file qwen3_8b_log.jsonl --max_job_send_time 10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Profile command with subcommands
    profile_parser = subparsers.add_parser('profile', help='Profile performance')
    profile_subparsers = profile_parser.add_subparsers(dest='profile_type', help='Profile type')
    
    # Prefill profiling - reuse existing argument parsers
    prefill_parser = profile_subparsers.add_parser('prefill', help='Profile prefill performance')
    ServerArgs.add_cli_args(prefill_parser)
    BenchArgs.add_cli_args(prefill_parser)
    
    # Decode profiling - reuse existing argument parsers
    decode_parser = profile_subparsers.add_parser('decode', help='Profile decode performance')
    ServerArgs.add_cli_args(decode_parser)
    BenchArgs.add_cli_args(decode_parser)
    
    # Real workload profiling
    real_parser = subparsers.add_parser('profile_real', help='Profile real workload using monkey patching')
    real_parser.add_argument('--model', type=str, required=True, help='Model name/path')
    real_parser.add_argument('--output_file', type=str, help='Output file for profiling results')
    real_parser.add_argument('--max_job_send_time', type=int, default=60, help='Maximum job send time in seconds')
    real_parser.add_argument('--data_file', type=str, default='data/splitwise_code.csv', help='Data file to use')
    real_parser.add_argument('--max_rps', type=int, default=10, help='Maximum requests per second')
    real_parser.add_argument('--rps_scale', type=float, default=1.0, help='RPS scaling factor')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Validate script paths exist for profile_real
    if args.command == 'profile_real' and not MONKEY_PATCH_SCRIPT.exists():
        print(f"Error: Monkey patch script not found at {MONKEY_PATCH_SCRIPT}")
        return 1
    
    # Execute the appropriate command
    if args.command == 'profile':
        if args.profile_type == 'prefill':
            server_args = ServerArgs.from_cli_args(args)
            bench_args = BenchArgs.from_cli_args(args)
            profile_prefill(server_args, bench_args)
        elif args.profile_type == 'decode':
            server_args = ServerArgs.from_cli_args(args)
            bench_args = BenchArgs.from_cli_args(args)
            profile_decode(server_args, bench_args)
        else:
            profile_parser.print_help()
            return 1
    elif args.command == 'profile_real':
        result = profile_real(args)
        return result.returncode if result else 0
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    main()