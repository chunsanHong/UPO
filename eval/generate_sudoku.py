import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
import pdb
# from accelerate import Accelerator
# acc = Accelerator()                    

# def distributed_breakpoint():
#     if acc.local_process_index == 0: 
#         pdb.set_trace()
#     acc.wait_for_everyone()  

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    guidance_model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    guidance_scale = 0.0,
    extra_temperature = 0,
    verbose = True,
    bim_size = 5,
    tau = 0.05,
    seed= 42,
):
    """
    Optimized version of the generate function.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = prompt.clone()
        prompt_index = x != mask_id
        total_steps = (x == mask_id).sum().item()
        
        for i in range(total_steps):
            mask_index = x == mask_id
            
            # Handle classifier-free guidance more efficiently
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)

                # Get logits in a single forward pass
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                current_bim = torch.ones_like(x)
                # current_bim[:, start_idx:end_idx] = 1
                current_mask = current_bim * mask_index
                raw_output = model(x, current_bim = current_bim, current_mask = current_mask, mode = 'get_extra')
                logits = raw_output.logits.clone()

            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Handle remasking strategy
            if remasking == "low_confidence":
                # Use float32 instead of float64 for better performance
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "margin":
                p = F.softmax(logits, dim=-1)
                x0_p = (p.sort(dim=-1).values[:,:,-1]-p.sort(dim=-1).values[:,:,-2])
            elif remasking == "entropy":
                p = F.softmax(logits, dim=-1)
                x0_p = ((torch.log(p)*p).sum(axis=-1))
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=x0.device)
            elif remasking in ['extra_head','extra_bim']:
                x0_p = raw_output.out_extra.clone()[:,:logits.shape[1]]
                x0_p = add_gumbel_noise(x0_p, temperature=extra_temperature)
                if x0_p.shape[1] == 32:
                    x0_p_ = torch.zeros_like(x0).to(x0_p.dtype)
                    x0_p_[:,start_idx:end_idx] =x0_p
                    x0_p = x0_p_
            elif remasking == 'bim':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                vals, indices = torch.topk(confidence, min((confidence[0] != -float('inf')).sum(), bim_size), dim=1)       # k 번째로 큰 값까지
                new_conf = torch.ones_like(x0, device = x0.device)*-np.inf
                new_conf.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=confidence.dtype, device = new_conf.device))
                x0_p = torch.rand(x0.shape, device=x0.device)
                confidence = x0_p*new_conf
            elif remasking == 'softmax_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                x0 = torch.where(mask_index, x0, x)
                bim_index = mask_index.clone()
                confidence = torch.exp(torch.where(bim_index.unsqueeze(-1).repeat(1,1,p.shape[2]),p,-np.inf)/tau)
                confidence = (confidence.sum(axis=-1,keepdim=True)/confidence.sum(axis=[1,2],keepdim=True)).squeeze(-1)
            else:
                raise NotImplementedError(remasking)

            # Update masked tokens
            x0 = torch.where(mask_index, x0, x)
            if remasking != 'softmax_confidence':
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            
            # Select tokens to transfer based on confidence
            for j in range(confidence.shape[0]):
                num_tokens = 1
                if num_tokens > 0:
                    if remasking == 'softmax_confidence':
                        select_indices = torch.multinomial(confidence[j], num_samples=1, replacement = False)
                        transfer_index[j, select_indices] = True
                    elif remasking != 'extra_bim':
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        transfer_index[j, select_indices] = True
                    else:
                        tmp_prob = F.softmax(raw_output.logits[j], dim = -1).max(axis=1).values
                        tmp_prob = (confidence[j] != -np.inf).float() * tmp_prob
                        topk_indices = torch.topk(tmp_prob, k = bim_size).indices
                        tmp_mask = torch.zeros_like(confidence[j])
                        tmp_mask[topk_indices] = 1
                        tmp_mask = tmp_mask.to(torch.bool)
                        tmp_confidence = confidence[j].clone()
                        tmp_confidence[~tmp_mask] = -np.inf
                        _, select_indices = torch.topk(tmp_confidence, k=num_tokens)
                        transfer_index[j, select_indices] = True
                        
            x[transfer_index] = x0[transfer_index]
        return x