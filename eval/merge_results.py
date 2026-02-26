import json
import re
import os
import glob
import tiktoken
from collections import defaultdict
import pandas as pd
import numpy as np
import argparse

def parse_sudoku_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data
    total_accuracy = []

    for item in data.get("generations", []):
        accuracy = item.get("accuracy", "")
        total_accuracy.append(accuracy)
    return total_accuracy


def extract_setup_name(filename):
    """Extract the setup name from the filename."""
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    if match:
        return match.group(1)
    return None


def aggregate_results(directory="."):
    """Aggregate results from all JSON files and save detailed results."""
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(directory, "*_generations.json"))

    merged_total_accuracy = []

    for json_file in json_files:
        filename = os.path.basename(json_file)
        setup_name = extract_setup_name(filename)

        if setup_name:
            # print(f"Processing {filename}...")
            if "sudoku" in setup_name:
                total_accuracy = parse_sudoku_answers(json_path=json_file)
                merged_total_accuracy += total_accuracy

    print('Total Accuracy:')
    print(sum(merged_total_accuracy)/len(merged_total_accuracy))


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate evaluation results")
    p.add_argument(
        "--result_path",
        type=str,
        default="eval/sudoku_test",
        help="Path to the evaluation results directory (e.g., eval/sudoku_test)",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    aggregate_results(directory=args.result_path)
