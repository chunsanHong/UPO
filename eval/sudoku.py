import re
import pandas as pd
from datasets import Dataset as HFDataset
import os
from parsers import Parser

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
import random
import re
from datasets import load_dataset
from parsers import Parser, is_equiv
import torch.distributed as dist

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

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=GSM_SYSTEM_PROMPT,
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.load_test_dataset()
        self.create_few_shot_prompt()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test", cache_dir = '/scratch/slurm-user3/chunsan/cached_data')

    def create_prompt(self, input_text):
        # Format similar to your chat function
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + prompt}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input

    def load_few_shot_examples(self):
        if isinstance(self.dataset, GSM8KDataset):
            train_data = load_dataset("gsm8k", "main", split="train")
            examples = random.sample(range(len(train_data)), self.num_examples)
            return [train_data[example] for example in examples]
        else:
            return []

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples"""
        few_shot_examples = self.load_few_shot_examples()

        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            answer = example["answer"]
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")
        self.few_shot_prompt = "\n\n".join(formatted_examples)

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["question"]
        answer = Parser.extract_answer_gsm8k(self.dataset[self.subsample[idx].item()]["answer"])
        prompt = self.create_prompt(question)
        return prompt, question, answer

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}



class SudokuDataset(GSM8KDataset):

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=SUDOKU_SYSTEM_PROMPT,
        subsample=256,
    ):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.sudoku_file_path = f"{cur_path}/../dataset/4x4_test_sudoku.csv"
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        """Load the Sudoku dataset from the CSV file."""
        df = pd.read_csv(self.sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
        # Convert pandas DataFrame to HuggingFace Dataset using from_pandas
        self.dataset = HFDataset.from_pandas(df)
        print("Loaded Testing Sudoku dataset with {} examples".format(len(self.dataset)))

    def format_sudoku_grid(self, sudoku_str):
        """Simplified function to format a sudoku string."""
        # Simply pass through the raw string as requested
        return sudoku_str
    
    def formatting_sudoku(self, puzzle):
        entered_sudoku = f"\n{puzzle[:4]}\n{puzzle[4:8]}\n{puzzle[8:12]}\n{puzzle[12:]}\n".replace('0', ' ')
        masked_sudoku = entered_sudoku.replace(' ','<|mdm_mask|>')
        formatted_sudoku = f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {entered_sudoku}\n"
        formatted_answer = f"\n<answer>{masked_sudoku}</answer><|eot_id|>"
        return formatted_sudoku, formatted_answer
    
    def create_prompt(self, input_text):
        # Format similar to your chat function
        messages = [{"role": "user", "content": self.formatting_sudoku(input_text)}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return user_input

    def validate_sudoku(self, solution_str, ground_truth=None, question=None):
        if len(question) == 16:
            puzzle_str = question
        else:
            match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
            if match:
                puzzle_str = match.group(1)
        empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
        empty_cells = len(empty_indices)
        print(f"Empty cells: {empty_cells}")
        print(puzzle_str)
        if solution_str is None or len(solution_str) == 0:
            return 0, empty_cells, 0.0

        # Handle length issues
        if len(solution_str) < 16:
            # Pad with zeros if too short
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            # Truncate if too long
            solution_str = solution_str[:16]

        assert len(puzzle_str) == 16
        # Count correct cells among originally empty cells
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        accuracy = correct_cells / empty_cells
        return correct_cells, empty_cells, accuracy

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        puzzle = self.dataset[self.subsample[idx].item()]["Puzzle"]
        solution = self.dataset[self.subsample[idx].item()]["Solution"]

        # Modified question format to reference the examples in the system prompt

        assert len(puzzle) == 16, f"Invalid puzzle length: {len(puzzle)}"

        prompt = self.create_prompt(puzzle)
        return prompt, puzzle, solution

def extract_answer_sudoku(solution_str):
    solution_str = solution_str.replace('\n', '')
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None

def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0