from __future__ import annotations

import argparse
import json
from pathlib import Path

from iho.pipeline import generate, load_model


DEFAULT_CHECKPOINT = "SEML-Lab/IHO-Llama-3-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate IHO attack prompts for a target affirmative response."
    )

    parser.add_argument(
        "target_response",
        nargs="?",
        default=None,
        help="Target affirmative response to generate attacks for.",
    )

    parser.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help="Optional text file containing the target affirmative response.",
    )

    parser.add_argument(
        "-n",
        "--num-attacks",
        type=int,
        default=32,
        help="Number of attacks to generate.",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Generation batch size. Use 1 on CPU/low VRAM.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for deterministic output.",
    )

    parser.add_argument(
        "--attack_size",
        type=int,
        default=32,
        help="Number of denoising/generation attack_steps.",
    )


    parser.add_argument(
        "--attack_steps",
        type=int,
        default=32,
        help="Number of denoising/generation attack_steps.",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="IHO checkpoint to load.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use.",
    )

    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype.",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional writable Hugging Face cache directory.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Return detailed JSON output instead of plain attacks.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output file.",
    )

    return parser.parse_args()


def get_target_response(args: argparse.Namespace) -> str:
    if args.target_file is not None:
        if not args.target_file.exists():
            raise FileNotFoundError(f"Target file not found: {args.target_file}")
        text = args.target_file.read_text(encoding="utf-8").strip()
    elif args.target_response is not None:
        text = args.target_response.strip()
    else:
        text = input("Target affirmative response: ").strip()

    if not text:
        raise ValueError("Target affirmative response cannot be empty.")

    return text


def main() -> None:
    args = parse_args()
    target_response = get_target_response(args)

    print(f"Loading checkpoint: {args.checkpoint}")

    model = load_model(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        attack_size=args.attack_size,
        attack_steps=args.attack_steps,
        temperature=args.temperature,
    )

    print(
        f"Generating {args.num_attacks} attack(s) "
        f"with batch_size={args.batch_size}, temperature={args.temperature}, attack_steps={args.attack_steps}..."
    )

    attacks = generate(
        model,
        target_response,
        num_attacks=args.num_attacks,
        batch_size=args.batch_size,
        temperature=args.temperature,
        attack_size=args.attack_size,
        attack_steps=args.attack_steps,
        return_dict=args.json,
    )

    if args.json:
        output_text = json.dumps(attacks, indent=2, ensure_ascii=False)
    else:
        output_text = "\n\n".join(
            f"=== Attack {i} ===\n{attack}"
            for i, attack in enumerate(attacks, start=1)
        )

    print()
    print(output_text)

    if args.output is not None:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"\nSaved output to: {args.output}")


if __name__ == "__main__":
    main()