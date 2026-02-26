import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from transformers.utils.import_utils import _is_package_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb
import pdb
import os
import inspect
from transformers.utils import is_sagemaker_dp_enabled, is_sagemaker_mp_enabled

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class UPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
        task: Optional[str] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.task = task

    def distributed_breakpoint(self):
        if self.accelerator.local_process_index == 0: 
            pdb.set_trace()
        self.accelerator.wait_for_everyone()  

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        inputs_extra_head, trajectory = inputs["inputs_extra_head"], inputs["trajectory"]
        trajectory_mask = inputs["trajectory_mask"]
        completion_ids = inputs["completion_ids"]
        sel_places = inputs["sel_places"]
        topk_and_mask = inputs["topk_and_mask"]
        trajectory_topk = inputs["trajectory_topk"]

        old_ps = inputs['old_ps']
        log_ref_probs = inputs['log_ref_probs']

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        # this_itr_mask_seed = mask_seeds[this_itr_idx]
        temp_sel = sel_places[this_itr_idx]
        input_ids = input_ids.unsqueeze(0)

        temp_inputs = self._parse_inputs(temp_sel, inputs_extra_head, topk_and_mask, trajectory, trajectory_topk, trajectory_mask)
        per_token_logps = self._get_extra_logps(model, **temp_inputs).reshape(len(prompt_ids),-1)
        # Compute the KL divergence between the model and the reference model
        

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"][this_itr_idx]
        coef_1 = torch.exp(torch.clamp(per_token_logps - old_per_token_logps, -10.0, 10.0))
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            if self.args.divergence == 'KL':
                all_logp = per_token_logps.sum(axis=1).detach()
                stopgrad_KL = torch.clamp(torch.exp((all_logp-torch.log(old_ps))),0.5,2) * (1+(all_logp-log_ref_probs))
                per_token_kl = stopgrad_KL * per_token_logps.sum(axis=1)
            elif self.args.divergence == 'CE':
                per_token_kl = self._get_CE(model, **temp_inputs).reshape(per_token_loss.shape[0],-1).sum(axis=1)

        loss = per_token_loss.sum(axis=1)
        
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = per_token_kl
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        if self.beta != 0.0:
            loss = loss + self.beta * per_token_kl

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = is_clipped.mean()

        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        del temp_inputs
        torch.cuda.empty_cache()

        return loss.mean() 

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def generate_sudoku(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        bim = -1,
        temperature = 0,
        extra_bim_size = 5,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = prompt.clone()

            prompt_index = x != mask_id

            total_steps = (x[0] == mask_id).sum().item()
            # Adjust steps if needed

            trajectory = [[] for _ in range(x.shape[0])]
            inputs_extra_head = []
            trajectory_mask = []
            trajectory_topk = [[] for _ in range(x.shape[0])]
            all_max_logp = []
            all_topk_and_mask = []
            all_log_ref_probs = torch.zeros(bs, dtype = torch.bfloat16, device = x.device)

            start_idx = prompt.shape[1] - block_length
            end_idx = prompt.shape[1]
            block_mask_index = x[:, start_idx:end_idx] == mask_id

            for i in range(total_steps):
                torch.cuda.empty_cache()
                mask_index = x == mask_id

                if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                    with torch.cuda.amp.autocast(enabled=self.args.fp16):
                        current_bim = torch.zeros_like(x)
                        current_bim[:, start_idx:end_idx] = 1
                        current_mask = current_bim * mask_index
                        trajectory_mask.append(current_mask)
                        raw_output = model(x, current_bim = current_bim, current_mask = current_mask, mode = 'get_extra')
                        logits = raw_output.logits.clone()
                        x0 = torch.argmax(logits, dim=-1)

                        inputs_extra_head.append(raw_output.processed_feats['x'])
                        all_topk_and_mask.append(raw_output.processed_feats['topk_and_mask'])

                        MDM_p = F.softmax(logits.to(dtype), dim=-1)
                        MDM_x0_p = torch.squeeze(
                            torch.gather(MDM_p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                        )
                        MDM_x0_p[:, end_idx:] = -np.inf
                        MDM_confidence = torch.where(mask_index, MDM_x0_p, -np.inf)
                            
                    
                        x0_p = raw_output.out_extra.clone()
                        x0_p = self.add_gumbel_noise(x0_p, temperature=temperature, dtype=dtype)
                        x0_p[:, end_idx:] = -np.inf
                        x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, -np.inf)

                        # Select tokens to transfer based on confidence
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        for j in range(confidence.shape[0]):
                            if self.args.dataset == 'sudoku':
                                num_tokens = 1
                            if remasking == 'extra_head':
                                _, select_indices = torch.topk(confidence[j], k=num_tokens)
                                transfer_index[j, select_indices] = True
                                sel_probs = MDM_p[j,select_indices]
                                mask_logits = MDM_p[j][mask_index[j]]
                                sel_indices = (mask_index[j].to(torch.int64).cumsum(dim=-1)*transfer_index[j].to(torch.int64)).amax()-1
                                ref_probs = (torch.exp(mask_logits/self.args.extra_temperature).sum(axis=1, keepdim=True)/torch.exp(mask_logits/self.args.extra_temperature).sum(axis=[0,1], keepdim = True)).squeeze(-1)
                                log_ref_probs = torch.log(ref_probs)
                                log_ref_probs = log_ref_probs[sel_indices]
                            elif remasking == 'extra_bim':
                                tmp_MDM_prob = (confidence[j] != -np.inf).float() * MDM_p[j].max(axis=1).values
                                if (confidence[j] != -float('inf')).sum() >= extra_bim_size:
                                    topk_indices = torch.topk(tmp_MDM_prob, k = extra_bim_size).indices
                                else:
                                    topk_indices = torch.topk(tmp_MDM_prob, k = (confidence[j] != -float('inf')).sum()).indices
                                    topk_indices = F.pad(topk_indices, (0, extra_bim_size - topk_indices.shape[0]), value = 0)
                                trajectory_topk[j].append(topk_indices.tolist())
                                
                                tmp_mask = torch.zeros_like(confidence[j])
                                tmp_mask[topk_indices] = 1
                                tmp_mask = tmp_mask.to(torch.bool)
                                tmp_confidence = confidence[j].clone()
                                tmp_confidence[~tmp_mask] = -np.inf
                                _, select_indices = torch.topk(tmp_confidence, k=num_tokens)
                                transfer_index[j, select_indices] = True
                                log_ref_probs = num_tokens*torch.log(torch.tensor([1/min((confidence!=-np.inf)[j].sum().item(), extra_bim_size)])).squeeze()
                            all_log_ref_probs[j] += log_ref_probs.to(all_log_ref_probs)
                        
                        x[transfer_index] = x0[transfer_index]
                            
                        trj_seq_idx, trj_token_idx = torch.where(transfer_index == True)
                        for tmp_idx in range(x.shape[0]):
                            trajectory[tmp_idx].append(trj_token_idx[torch.where(trj_seq_idx == tmp_idx)[0]].tolist())
                        
                        del x0, confidence, transfer_index, trj_seq_idx, trj_token_idx
        torch.cuda.empty_cache()
        return x, torch.stack(inputs_extra_head).permute(1,0,2,3), trajectory, torch.stack(trajectory_mask).permute(1,0,2), torch.stack(all_topk_and_mask).permute(1,0,2,3), trajectory_topk, all_log_ref_probs

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    def _parse_inputs(
        self, sel_indices, inputs_extra_head, topk_and_mask, trajectory, trajectory_topk, trajectory_mask,
    ):
        inputs_extra_head = {'x': inputs_extra_head[:,:, sel_indices].flatten(0,2), 'attention_bias': None, 'layer_past': None, 'use_cache': None, 'topk_and_mask': topk_and_mask[:,:,sel_indices].flatten(0,2)}
        trajectory = trajectory[:,:,sel_indices].flatten(0,2)
        if self.args.remasking == 'extra_head':
            candidates = trajectory_mask[:,:,sel_indices].flatten(0,2)
        elif self.args.remasking == 'extra_bim':
            trajectory_mask = trajectory_mask[:,:,sel_indices].flatten(0,2)
            trajectory_topk = trajectory_topk[:,:,sel_indices].flatten(0,2)
            candidates = torch.zeros_like(trajectory_mask)
            candidates.scatter_(dim=1, index=trajectory_topk, src=torch.ones_like(trajectory_topk, dtype=trajectory_topk.dtype))
            candidates[:,0] = 0
        return {'inputs_extra_head':inputs_extra_head, 'trajectory': trajectory, 'candidates': candidates}

    def _get_extra_logps(self, model, inputs_extra_head, trajectory, candidates):
        extra_out = model(inputs_extra_head, mode = 'get_extra_only')
        logp_extra = (extra_out - torch.log((torch.exp(extra_out.float())*candidates.float()).sum(axis=1, keepdim = True).clamp_min(1e-12))).gather(dim=1, index = trajectory)
        logp_extra_sum = logp_extra.sum(axis=1)
        logp_extra_sum = logp_extra_sum.to(extra_out.dtype)
        return logp_extra_sum

    def _get_CE(self, model, inputs_extra_head, candidates = None, **kwargs):
        def max_tie_normalized_onehot(x: torch.Tensor, eps: float = 0.0):
            max_vals = x.max(dim=1, keepdim=True).values             
            mask = x == max_vals                                     
            counts = mask.sum(dim=1, keepdim=True)                   
            return mask.to(x.dtype) / counts.clamp(min=1)   
        out_extra = model(inputs_extra_head, mode = 'get_extra_only')
        logp_extra = (out_extra.float() - torch.log((torch.exp(out_extra.float())*candidates.float()).sum(axis=1, keepdim = True).clamp_min(1e-12)))
        target = max_tie_normalized_onehot(inputs_extra_head['topk_and_mask'][:,:,2].float()*candidates.float())
        kld = - (target*logp_extra).sum(axis=1)
        return kld

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        
        if self.task == 'sudoku':
            prompts_text = [
                maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
            ]
            prompts_text = [text + x["answer_template"] for (text, x) in zip(prompts_text, inputs)]
        
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            inputs_extra_head_all = []
            trajectory_all = []
            trajectory_mask_all = []
            topk_and_mask_all = []
            log_ref_probs_all = []
            trajectory_topk_all = []
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                # WARNING: Attention masks are not currently used during generation.
                # This works fine as we set num_generations == per_device_train_batch_size (no padding tokens created) in our config, but may cause
                # unintended attention to padding tokens when num_generations is smaller.
                # As currently we find Llada's modeling file does not handle attention mask. We will address this in future update soon.
                if self.task == 'sudoku':
                    batch_prompt_completion_ids, inputs_extra_head, trajectory, trajectory_mask, topk_and_mask, trajectory_topk, log_ref_probs = self.generate_sudoku(
                        model=unwrapped_model,
                        prompt=batch_prompt_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        cfg_scale=cfg_scale,
                        remasking = self.args.remasking,
                        mask_id=self.args.mask_id,
                        temperature = temperature, 
                        extra_bim_size = self.args.extra_bim_size,
                    )

                prompt_completion_ids_all.append(batch_prompt_completion_ids)
                inputs_extra_head_all.append(inputs_extra_head)
                trajectory_all.append(trajectory)
                trajectory_mask_all.append(trajectory_mask)
                topk_and_mask_all.append(topk_and_mask)
                trajectory_topk_all.append(trajectory_topk)
                log_ref_probs_all.append(log_ref_probs)

                del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids, trajectory_mask, topk_and_mask
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)
            inputs_extra_head = torch.stack(inputs_extra_head_all, dim=0).to(device)
            topk_and_mask = torch.stack(topk_and_mask_all, dim=0)
            trajectory = torch.tensor(trajectory_all).to(device)
            trajectory_mask = torch.stack(trajectory_mask_all).to(device)
            trajectory_topk = torch.tensor(trajectory_topk_all).to(device)
            log_ref_probs = torch.cat(log_ref_probs_all)
            
            del trajectory_all, trajectory_mask_all, inputs_extra_head_all, topk_and_mask_all, log_ref_probs_all
            torch.cuda.empty_cache()

        if self.task in ['sudoku']:
            prompts_text = [
                maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
            ]
            prompts_text = [text for (text, x) in zip(prompts_text, inputs)]
            prompt_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        all_old_per_token_logps = []
        all_ref_per_token_logps = []

        extra_batch_size = 8

        with torch.no_grad():
            if self.num_iterations > 1:
                if self.task == 'sudoku':
                    sel_places = torch.tensor([list(range(inputs_extra_head.shape[2])) for _ in range(self.num_iterations)], dtype = torch.long, device = inputs_extra_head.device)

                all_old_per_token_logps = []
                for temp_sel in sel_places:
                    temp_inputs = self._parse_inputs(temp_sel, inputs_extra_head, topk_and_mask, trajectory, trajectory_topk, trajectory_mask)
                    old_per_token_logps = self._get_extra_logps(self.model, **temp_inputs).reshape(len(prompts),-1)
                    all_old_per_token_logps.append(old_per_token_logps)
                    del temp_inputs
                
                all_old_per_token_logps = torch.stack(all_old_per_token_logps)
                if self.task == 'sudoku':
                    old_ps = torch.exp(all_old_per_token_logps[0].sum(axis=1))
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                all_ref_per_token_logps = all_old_per_token_logps.clone()

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if self.task == 'sudoku': completions_text = [text.replace('\n','').replace(' ','0') for text in completions_text]
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.norm_advantage: advantages = advantages/(std_grouped_rewards+1e-4)*((std_grouped_rewards!=0).float())
        
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                # if is_rich_available():
                if _is_package_available("rich"):
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "inputs_extra_head": inputs_extra_head,
            "trajectory": trajectory,
            "trajectory_mask": trajectory_mask,
            "trajectory_topk": trajectory_topk,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "sel_places": sel_places,
            "topk_and_mask": topk_and_mask,
            "old_ps": old_ps,
            "log_ref_probs": log_ref_probs
        }
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # logger.info(f"Saving model checkpoint to {output_dir}")

        torch.save({k:v for (k,v) in state_dict.items() if 'extra_head' in k}, os.path.join(output_dir, 'extra_head.pt'))
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer