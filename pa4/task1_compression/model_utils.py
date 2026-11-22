"""Loading the fp16 checkpoint, listing eligible matrices, restoring weights and counting bytes."""
from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import get_device


def load_model(name):
    device=get_device()
    if device.type=="cuda":
        dtype=torch.float16 #manual: fp16 checkpoint, t4 has no bf16
    else:
        dtype=torch.float32 #cpu tests only
    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name,dtype=dtype,attn_implementation="sdpa")
    model=model.to(device)
    model.eval()
    return model,tokenizer


def eligible_linears(model,names):
    """Every eligible projection in every block, in block order. Returns a list of (full_name,block_index,short_name,module)."""
    out=[]
    layers=model.model.layers
    for block_index in range(len(layers)):
        layer=layers[block_index]
        for short in names:
            if short in ["q_proj","k_proj","v_proj","o_proj"]:
                module=getattr(layer.self_attn,short) #attention projections
                full=f"model.layers.{block_index}.self_attn.{short}"
            else:
                module=getattr(layer.mlp,short) #mlp projections
                full=f"model.layers.{block_index}.mlp.{short}"
            out.append((full,block_index,short,module))
    return out


def backup_state(model):
    """Cpu copy of every parameter, used to restore the untouched fp16 model before each condition."""
    backup={}
    for name,tensor in model.state_dict().items():
        backup[name]=tensor.detach().to("cpu").clone()
    return backup


def restore_state(model,backup):
    """Manual: each condition starts from an independent copy of the unchanged fp16 weights."""
    remove_input_scales(model) #first drop awq o_proj scales, they are not in the backup
    model.load_state_dict(backup,strict=True)


def count_eligible(entries):
    total=0
    for full,block_index,short,module in entries:
        total=total+module.weight.numel()
    return total


def unique_parameters(model):
    """Parameters counted once even when shared (qwen 1.5b ties embed_tokens and lm_head)."""
    seen=set()
    out=[]
    for name,parameter in model.named_parameters(remove_duplicate=False):
        pointer=parameter.data_ptr()
        if pointer in seen:
            continue #tied copy, already counted
        seen.add(pointer)
        out.append((name,parameter))
    return out


def fp16_bytes_outside(model,entries):
    """Bytes of everything that stays fp16: embeddings, norms, biases, lm head (counted once if tied)."""
    eligible_pointers=set()
    for full,block_index,short,module in entries:
        eligible_pointers.add(module.weight.data_ptr())
    total=0
    for name,parameter in unique_parameters(model):
        if parameter.data_ptr() in eligible_pointers:
            continue #eligible weights are counted by the method's own format
        total=total+parameter.numel()*2 #2 bytes per fp16 value
    return total


def quant_bytes_matrix(rows,columns,bits,group_size):
    """Packed bytes for one matrix with group-wise asymmetric quantization along input columns.
    Per group: group_size*bits packed weight bits, a 16-bit fp16 scale and a bits-wide integer zero-point.
    Columns are padded up to a multiple of group_size (padding bits are counted)."""
    groups=math.ceil(columns/group_size) #per row
    padded=groups*group_size
    weight_bits=rows*padded*bits
    meta_bits=rows*groups*(16+bits) #scale fp16 + zero-point at the weight width
    return math.ceil((weight_bits+meta_bits)/8)


def sparse_bytes_matrix(rows,columns,nonzero):
    """Manual's unstructured 16-bit sparse format: one mask bit per weight plus 16 bits per nonzero."""
    return math.ceil((rows*columns+16*nonzero)/8)


def nm_bytes_matrix(rows,columns,n,m):
    """Ours, n:m semi-structured (like 2:4 on ampere): each kept value fp16 plus a 2-bit index inside its group of 4."""
    kept=rows*(columns//m)*n
    index_bits=math.ceil(math.log2(m)) #2 bits for m=4
    return math.ceil(kept*(16+index_bits)/8)


class InputScale(torch.nn.Module):
    """Keeps 1/s for an awq channel scale that could not be folded (o_proj). Stored as fp16 and counted in bytes."""

    def __init__(self,scale):
        super().__init__()
        self.register_buffer("inverse",(1.0/scale).to(torch.float16)) #x is divided by s before the linear


def attach_input_scale(module,scale):
    holder=InputScale(scale).to(module.weight.device)
    module.awq_input_scale=holder #kept on the module so the hook can find it

    def hook(mod,args):
        x=args[0]
        return (x*mod.awq_input_scale.inverse.to(x.dtype),) #divide input channels by s

    handle=module.register_forward_pre_hook(hook)
    module.awq_hook_handle=handle


def remove_input_scales(model):
    for module in model.modules():
        if hasattr(module,"awq_hook_handle"):
            module.awq_hook_handle.remove()
            del module.awq_hook_handle
            del module.awq_input_scale


def input_scale_bytes(model):
    total=0
    for module in model.modules():
        if hasattr(module,"awq_input_scale"):
            total=total+module.awq_input_scale.inverse.numel()*2 #fp16
    return total
