import dataclasses
import argparse
from typing import Tuple


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_filename_prefix: str = "profile"
    batch_composition: list = dataclasses.field(default_factory=list)
    run_prefill_profiling: bool = False
    run_decode_profiling: bool = False
    run_prefill_profiling_with_prefix_cache: bool = False
    max_decode_token_length: int = int(1e9)
    chunk_prefill: bool = False
    chunk_size: int = 512
    skew: str = "none"
    output_dir: str = "profiling_results"
    max_batch_size: int = 256
    max_input_len: int = 16384

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument(
            "--profile", action="store_true", help="Use Torch Profiler."
        )
        parser.add_argument(
            "--run-prefill-profiling",
            action="store_true",
            help="Run prefill profiling.",
        )
        parser.add_argument(
            "--run-decode-profiling", action="store_true", help="Run decode profiling."
        )
        parser.add_argument(
            "--max-decode-token-length",
            type=int,
            default=BenchArgs.max_decode_token_length,
            help="Maximum token length to consider for decode profiling.",
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--chunk-prefill",
            action="store_true",
            help="Split prefill across multiple requests",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=BenchArgs.chunk_size,
            help="Size of each prefill chunk (default: 512)",
        )
        parser.add_argument(
            "--skew",
            type=str,
            choices=["none", "medium", "heavy"],
            default=BenchArgs.skew,
            help="Batch length distribution : none (uniform), medium (mild skew), heavy (noticeable skew)",
        )
        parser.add_argument(
            "--run-prefill-profiling-with-prefix-cache",
            action="store_true",
            help="Run prefill profiling with prefix cache.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=BenchArgs.output_dir,
            help="Output directory for profiling results.",
        )
        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=BenchArgs.max_batch_size,
            help="Maximum batch size for profiling.",
        )
        parser.add_argument(
            "--max-input-len",
            type=int,
            default=BenchArgs.max_input_len,
            help="Maximum input length for profiling.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "BenchArgs":
        # use the default value's type to cast the args into correct types.
        kwargs = {}
        for field in dataclasses.fields(cls):
            if hasattr(args, field.name):
                value = getattr(args, field.name)
                # Handle special cases where type conversion might fail
                if field.name in ['batch_size', 'input_len', 'output_len']:
                    # These should be tuples, convert list to tuple if needed
                    if isinstance(value, list):
                        kwargs[field.name] = tuple(value)
                    elif isinstance(value, tuple):
                        kwargs[field.name] = value
                    else:
                        kwargs[field.name] = (value,)
                elif field.name == 'batch_composition' and value is None:
                    kwargs[field.name] = []
                else:
                    # For simple types, use direct assignment
                    kwargs[field.name] = value
        return cls(**kwargs)
