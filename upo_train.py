import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig
import copy

from upo_trainer import UPOTrainer
from upo_config import UPOConfig
from reward_func import (
    sudoku_reward_func,
)
from data_utils import (
    get_sudoku_questions,
    set_random_seed,
)

from modeling_llada import LLaDAModelLM
from torch.optim import AdamW

def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset based on configuration
    if grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = LLaDAModelLM.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True,
    torch_dtype="bfloat16",
    cache_dir = '/scratch/slurm-user3/chunsan/cached_models',
    quantization_config = bnb_config,
    head_max_sequence_len = 1024, 
    head_num_feats = 10, 
    head_num_hidden_channels = 3
    ).to(device)
    
    EXTRA_HEAD_PATH = grpo_config.extra_model_path
    extra_state = torch.load(EXTRA_HEAD_PATH, map_location=device)
    model.model.extra_head.load_state_dict(extra_state, strict=True)

    for name, param in model.model.named_parameters():
        if "extra_head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.config.use_cache = False
    
    model.is_quantized = False
    
    trainer = UPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        task = grpo_config.dataset,
    )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((UPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
