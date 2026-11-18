"""Run the paper's MATH and GSM8K scoring pipeline with batched vLLM inference."""

import gzip
import json
import math
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path
from sys import float_info

from hypothesis_testing.scorers import score
from hypothesis_testing.testing import calibrate, evaluate
from uncertainty_estimation.gsm8k import ln_e_drop as gsm8k
from uncertainty_estimation.math import ln_e_drop as math_dataset
from uncertainty_estimation.utils.vllm_utils import (
    add_vllm_tp_args,
    llm_common_kwargs,
    setup_vllm_distributed_and_tp,
)


def read_split(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def math_reference(row):
    if "answer" in row:
        return str(row["answer"])
    return math_dataset.extract_boxed_answer(row["solution"])[0]


def score_split(rows, args, llm, tokenizer, sampling, risk):
    if args.dataset == "math":
        key = "problem"
        prompt = math_dataset.format_math_prompt
        reference = math_reference
        equivalent = math_dataset.is_math_equivalent
    else:
        key = "question"
        prompt = gsm8k.format_gsm8k_prompt
        reference = lambda row: gsm8k.extract_gsm8k_answer(row["answer"])
        equivalent = gsm8k.is_gsm8k_correct

    results = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        outputs = llm.generate([prompt(row[key]) for row in batch], sampling)
        for row, output in zip(batch, outputs, strict=True):
            completion = output.outputs[0]
            generated, logprobs = completion.text, completion.logprobs
            if args.dataset == "math":
                _, logprobs, predicted = math_dataset.truncate_response(
                    generated, logprobs, tokenizer
                )
            else:
                predicted = gsm8k.extract_gsm8k_answer(generated)
            gold = reference(row)
            # A completion without a token signal must abstain.
            value = float(risk(logprobs)) if logprobs and any(logprobs) else float_info.max
            if not math.isfinite(value):
                raise ValueError(f"Non-finite risk for {row[key]!r}")
            results.append({
                "question": row[key],
                "pred_answer": predicted,
                "gt_answer": gold,
                "is_correct": int(equivalent(predicted, gold)),
                "risk_score": value,
            })
    return results


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("math", "gsm8k"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Hugging Face model ID or local path")
    parser.add_argument("--method", default="shannon-drop", choices=(
        "ln-s", "ln-s-drop", "logtoku", "logtoku-drop", "shannon", "shannon-drop",
    ))
    parser.add_argument("--scorer", help="Optional custom risk callable in module:function form")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--ema-span", type=float, default=5)
    parser.add_argument("--drop-k", type=int, default=5)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    add_vllm_tp_args(parser, default_gpu_memory_utilization=0.85)
    args = parser.parse_args()
    if min(args.drop_k, args.logprobs, args.batch_size, args.max_tokens) < 1 or args.ema_span < 1:
        parser.error("drop-k, logprobs, batch-size, max-tokens and ema-span must be positive.")
    if not all(0 < value < 1 for value in (args.alpha, args.delta)):
        parser.error("alpha and delta must both be in (0, 1).")

    if args.scorer:
        module, separator, name = args.scorer.partition(":")
        if not separator or not module or not name:
            parser.error("--scorer must have the form module:function.")
        risk = getattr(import_module(module), name)
        if not callable(risk):
            parser.error("--scorer must reference a callable.")
    else:
        def risk(logprobs):
            return score(args.method, logprobs, ema_span=args.ema_span, drop_k=args.drop_k)

    data = args.data_dir / args.dataset
    calibration = read_split(data / "calibration.jsonl.gz")
    test = read_split(data / "test.jsonl.gz")
    key = "problem" if args.dataset == "math" else "question"
    if not calibration or not test:
        parser.error("Calibration and test must both contain examples.")
    if {row[key] for row in calibration} & {row[key] for row in test}:
        parser.error("Calibration and test contain overlapping questions.")

    from vllm import LLM, SamplingParams

    settings = llm_common_kwargs(args, setup_vllm_distributed_and_tp(args))
    settings.update(max_logprobs=args.logprobs, trust_remote_code=False)
    llm = LLM(**settings)
    tokenizer = llm.get_tokenizer()
    stops = ["Problem:", "Solution:"] if args.dataset == "math" else ["Question:", "Answer:"]
    sampling = SamplingParams(
        temperature=0, top_p=0.95, max_tokens=args.max_tokens,
        logprobs=args.logprobs, stop=stops,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_scores = score_split(calibration, args, llm, tokenizer, sampling, risk)
    write_json(args.output_dir / "calibration-scores.json", calibration_scores)
    threshold = calibrate(calibration_scores, args.alpha, args.delta)
    write_json(args.output_dir / "threshold.json", threshold)

    test_scores = score_split(test, args, llm, tokenizer, sampling, risk)
    write_json(args.output_dir / "test-scores.json", test_scores)
    metrics = evaluate(test_scores, threshold["threshold"])
    run_parameters = {name: value for name, value in vars(args).items() if not isinstance(value, Path)}
    write_json(args.output_dir / "metrics.json", {**run_parameters, **threshold, **metrics})
    print(json.dumps({**threshold, **metrics}, indent=2))


if __name__ == "__main__":
    main()
