## --------------------------------------------------------PART:Load question model code----------------------------------------------------------##
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        LlamaForCausalLM,
        AutoModel,
        AutoModelForSeq2SeqLM,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        LLamaForCausalLM,
        AutoModel,
        AutoModelForSeq2SeqLM,
    )
import warnings
from torch import Tensor
import dataclasses
from torch.nn import functional as F
import torch.nn as nn
import torch


@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""

    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True


default_compression_config = CompressionConfig(num_bits=8, group_size=256, group_dim=1, symmetric=True, enabled=True)


def raise_warning_for_old_weights(model_path, model):
    if "vicuna" in model_path.lower():
        try:
            is_vicuna = isinstance(model, LlamaForCausalLM)
        except Exception:
            is_vicuna = isinstance(model, LLamaForCausalLM)
        if is_vicuna and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fschat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
            )


def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = original_shape[:group_dim] + (num_groups, group_size) + original_shape[group_dim + 1 :]

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = original_shape[:group_dim] + (pad_len,) + original_shape[group_dim + 1 :]
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim,
        )
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2**num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size,
        config.num_bits,
        config.group_dim,
        config.symmetric,
    )

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim] + (original_shape[group_dim] + pad_len,) + original_shape[group_dim + 1 :]
        )
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


class CLinear(nn.Module):
    """Compressed Linear Layer."""

    def __init__(self, weight, bias, device):
        super().__init__()

        self.weight = compress(weight.data.to(device), default_compression_config)
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        weight = decompress(self.weight, default_compression_config)
        return F.linear(input, weight, self.bias)


def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = torch.cuda.device_count() if max_gpus is None else min(max_gpus, torch.cuda.device_count())

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def compress_module(module, target_device):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Linear:
            setattr(
                module,
                attr_str,
                CLinear(target_attr.weight, target_attr.bias, target_device),
            )
    for name, child in module.named_children():
        compress_module(child, target_device)


def load_model(model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        print("init_kwargs", kwargs)
    # elif device == "mps":
    #     kwargs = {"torch_dtype": torch.float16}
    #     # Avoid bugs in mps backend by not using in-place operations.
    #     replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs).cuda()
    elif "google/flan-t5" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    elif "dolly" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
    elif "pythia" in model_path or "stablelm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        raise_warning_for_old_weights(model_path, model)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


## --------------------------------------------------------PART:Load question model code---------------------------------------------------------- ##
