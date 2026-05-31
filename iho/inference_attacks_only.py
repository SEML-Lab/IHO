from __future__ import annotations

import argparse
import json
from pathlib import Path

from iho.pipeline import DEFAULT_BASE_MODEL, DEFAULT_CHECKPOINT, generate, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attack strings from an IHO attacker checkpoint.")
    parser.add_argument("target_response", help="Affirmative target response to inpaint attacks for.")
    parser.add_argument("--checkpoint", "--lora-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--num-attacks", "-n", type=int, default=1)
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--attack-size", type=int, default=32)
    parser.add_argument("--attack-steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        device=args.device,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        attack_size=args.attack_size,
    )
    attacks = generate(
        model,
        args.target_response,
        num_attacks=args.num_attacks,
        batch_size=args.batch_size,
        return_dict=args.json,
        attack_size=args.attack_size,
        attack_steps=args.attack_steps,
        temperature=args.temperature,
    )

    if args.json:
        print(json.dumps(attacks, indent=2))
    else:
        for attack in attacks:
            print(attack)


if __name__ == "__main__":
    main()
