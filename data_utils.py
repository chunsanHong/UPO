from datasets import load_dataset, Dataset
import pandas as pd
import random
import numpy as np
import torch
import os
import itertools
from nltk.lm import Vocabulary
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
import random
import re
from datasets import load_dataset
import torch.distributed as dist
import pickle

def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where ' ' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid. Never leave it as ' '.

Respond in this exact format:
<answer>
[First raw of 4-character solution]
[Second raw of 4-character solution]
[Third raw of 4-character solution]
[Firth raw of 4-character solution]
</answer>
"""

def formatting_sudoku(puzzle):
    entered_sudoku = f"\n{puzzle[:4]}\n{puzzle[4:8]}\n{puzzle[8:12]}\n{puzzle[12:]}\n".replace('0', ' ')
    masked_sudoku = entered_sudoku.replace(' ','<|mdm_mask|>')
    formatted_sudoku = f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {entered_sudoku}\n"
    formatted_answer = f"\n<answer>{masked_sudoku}</answer><|eot_id|>"
    return formatted_sudoku, formatted_answer

def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "/scratch/slurm-user3/chunsan/reproducible_upo/dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    # "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                    "content": formatting_sudoku(x['Puzzle'])[0]
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
            "answer_template": formatting_sudoku(x['Puzzle'])[1]
        }
    )

