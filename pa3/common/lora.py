from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


#own lora layer, used by task 1 (lora-4, lora-16, oplora) and task 2 (stages 2-3)
#written by hand so lora-16 and oplora share the exact same init and forward, oplora only adds the projection


class LoRALinear(nn.Module):
    def __init__(self,base,r,alpha,dropout,U_k=None,V_k=None):
        super().__init__()
        self.base=base #original frozen nn.Linear (fp16 backbone)
        self.r=r #adapter rank
        self.scale=alpha/r #alpha/r in W = W0 + (alpha/r) B A
        out_features=base.weight.shape[0]
        in_features=base.weight.shape[1]
        self.lora_A=nn.Parameter(torch.empty(r,in_features,dtype=torch.float32,device=base.weight.device)) #A in R^{r x d_in}, fp32 adapter
        self.lora_B=nn.Parameter(torch.zeros(out_features,r,dtype=torch.float32,device=base.weight.device)) #B in R^{d_out x r}, zero init so W = W0 at step 0
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5)) #random A, same rule peft uses
        self.dropout=nn.Dropout(dropout) #manual: dropout 0.05 on the adapter input
        #oplora bases, None for plain lora. buffers so they move with .to() but are not trained
        if U_k is not None:
            self.register_buffer("U_k",U_k.to(torch.float32),persistent=False) #leading left singular vectors of W0 [d_out, k]
            self.register_buffer("V_k",V_k.to(torch.float32),persistent=False) #leading right singular vectors of W0 [d_in, k]
        else:
            self.U_k=None
            self.V_k=None

    def effective_factors(self):
        """B~ and A~ for oplora (manual eq.), raw B and A for lora. Done in fp32 so the projection is exact."""
        A=self.lora_A
        B=self.lora_B
        if self.U_k is not None:
            with torch.autocast(device_type="cuda",enabled=False):
                B=B-self.U_k@(self.U_k.T@B) #B~ = B - U_k (U_k^T B), never builds the d_out x d_out projector
                A=A-(A@self.V_k)@self.V_k.T #A~ = A - (A V_k) V_k^T
        return A,B

    def delta_weight(self):
        """Dense Delta W = (alpha/r) B~ A~ in fp32, used for merging and diagnostics."""
        A,B=self.effective_factors()
        return self.scale*(B@A)

    def forward(self,x):
        result=self.base(x) #frozen W0 x
        A,B=self.effective_factors()
        lora_out=F.linear(F.linear(self.dropout(x),A),B) #B (A x), two thin matmuls instead of the dense update
        return result+self.scale*lora_out


def find_target_linears(model,target_names):
    """(full_name, parent_module, attribute_name, linear) for every nn.Linear whose last name is a target."""
    found=[]
    for full_name,module in model.named_modules():
        for child_name,child in module.named_children():
            if child_name in target_names and isinstance(child,nn.Linear):
                if full_name=="":
                    found.append((child_name,module,child_name,child))
                else:
                    found.append((full_name+"."+child_name,module,child_name,child))
    return found


def svd_bases(weight,k):
    """Leading k left and right singular vectors of W0, computed once in fp32 (manual)."""
    W=weight.detach().to(torch.float32)
    U,S,Vh=torch.linalg.svd(W,full_matrices=False) #W = U diag(S) Vh
    U_k=U[:,:k].contiguous() #[d_out, k]
    V_k=Vh[:k,:].T.contiguous() #[d_in, k]
    return U_k,V_k


def inject_lora(model,target_names,r,alpha,dropout,protect_rank=None):
    """Replace every target nn.Linear with LoRALinear. protect_rank=k turns it into oplora.

    Returns the list of replaced module names, in model order.
    """
    names=[]
    for full_name,parent,child_name,linear in find_target_linears(model,target_names):
        U_k=None
        V_k=None
        if protect_rank is not None:
            U_k,V_k=svd_bases(linear.weight,protect_rank) #bases of the starting weight
        setattr(parent,child_name,LoRALinear(linear,r,alpha,dropout,U_k,V_k))
        names.append(full_name)
    return names


def lora_modules(model):
    """name -> LoRALinear for every adapter in the model."""
    found={}
    for name,module in model.named_modules():
        if isinstance(module,LoRALinear):
            found[name]=module
    return found


def adapter_state(model,projected):
    """Adapter tensors to save. projected=True stores A~, B~ (what oplora exports for inference)."""
    state={}
    for name,module in lora_modules(model).items():
        if projected==True:
            A,B=module.effective_factors()
        else:
            A=module.lora_A
            B=module.lora_B
        state[name+".lora_A"]=A.detach().float().cpu().contiguous()
        state[name+".lora_B"]=B.detach().float().cpu().contiguous()
    return state


def merge_factors_into(model,factors,scale):
    """W <- W + scale * B A for every saved adapter, on a fresh backbone copy (in place)."""
    merged=[] #module names we merged
    named=dict(model.named_modules())
    for key in factors:
        if key.endswith(".lora_A"):
            name=key[:-len(".lora_A")]
            A=factors[name+".lora_A"].to(torch.float32)
            B=factors[name+".lora_B"].to(torch.float32)
            linear=named[name]
            delta=scale*(B@A) #dense update in fp32
            with torch.no_grad():
                new_weight=linear.weight.data.to(torch.float32)+delta.to(linear.weight.device)
                linear.weight.data.copy_(new_weight.to(linear.weight.dtype))
            merged.append(name)
    return merged
