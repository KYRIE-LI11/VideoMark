import json
from collections import defaultdict
import argparse


def calculate_average_scores(json_path):
    """Read JSON file and compute average scores per model."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Initialize stats dictionary for each model
    model_stats = defaultdict(lambda: {
        'subject_consistency': [],
        'background_consistency': [],
        'motion_smoothness': [],
        'imaging_quality': [],
        'total_score': []
    })

    # Iterate through each sample and collect scores
    for scene in data.values():
        for model, scores in scene.items():
            for dimension in model_stats[model].keys():
                value = scores.get(dimension)
                if value is not None:
                    model_stats[model][dimension].append(value)

    # Calculate averages
    averages = {}
    for model, dimensions in model_stats.items():
        averages[model] = {
            dimension: round(sum(values) / len(values), 2) if values else 0.0
            for dimension, values in dimensions.items()
        }

    return averages


def main():
    parser = argparse.ArgumentParser(description="Calculate average video scores per model")
    parser.add_argument(
        '--json-path', type=str, required=True,
        help='Path to JSON results file'
    )
    args = parser.parse_args()

    averages = calculate_average_scores(args.json_path)

    # Print header
    print("{:<12} {:<20} {:<20} {:<20} {:<20} {:<20}".format(
        "Model", "Subject Consistency", "Background Consistency",
        "Motion Smoothness", "Imaging Quality", "Total Score"
    ))
    # Print each model's averages
    for model, scores in sorted(averages.items()):
        print("{:<12} {:<20} {:<20} {:<20} {:<20} {:<20}".format(
            model,
            scores['subject_consistency'],
            scores['background_consistency'],
            scores['motion_smoothness'],
            scores['imaging_quality'],
            scores['total_score']
        ))


if __name__ == "__main__":
    main()