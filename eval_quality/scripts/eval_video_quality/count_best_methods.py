#!/usr/bin/env python3
import json
import argparse
from typing import Dict, Any

def main():
    parser = argparse.ArgumentParser(
        description="Count how many times each method achieves the highest total score per sample"
    )
    parser.add_argument(
        '--json-path', type=str, required=True,
        help='Path to the JSON file containing score data'
    )
    parser.add_argument(
        '--methods', nargs='+', default=[
            'revmark', 'rivagan', 'videoseal', 'videoshield', 'videomark', 'without_watermark'
        ],
        help='List of method names to consider'
    )
    args = parser.parse_args()

    # Load data
    with open(args.json_path, 'r') as f:
        data: Dict[str, Any] = json.load(f)

    # Initialize counts
    count: Dict[str, int] = {method: 0 for method in args.methods}

    # Iterate over each sample
    for sample, scores in data.items():
        try:
            # Extract total_score for each method
            totals = {
                method: scores[method]['total_score']
                for method in args.methods
            }
            max_score = max(totals.values())
            # Increment count for any method tied at max_score
            for method, score in totals.items():
                if score == max_score:
                    count[method] += 1
        except (KeyError, TypeError):
            # Skip samples with missing data
            continue

    # Print results
    print("Results:")
    for method, c in count.items():
        print(f"{method}: {c}")

if __name__ == '__main__':
    main()