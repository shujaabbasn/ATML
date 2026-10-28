from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


#the visual interface: connector g_phi, prompt building with visual tokens inside qwen's user message,
#masked losses, reference nll and a small greedy decoder that works on input embeddings

IMAGE_TOKEN="<|image_pad|>" #existing qwen special token, only used as a placeholder, its embedding is replaced


class Connector(nn.Module):
    """Linear 768 -> 1024, GELU, Linear 1024 -> 1024 (manual). pool_to_49 averages 2x2 blocks first (optional step 7)."""

    def __init__(self,vision_dim,hidden_dim,pool_to_49):
        super().__init__()
        self.fc1=nn.Linear(vision_dim,hidden_dim)
        self.act=nn.GELU()
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.pool_to_49=pool_to_49
        self.register_buffer("output_scale",torch.tensor(1.0)) #fixed calibration scale, set once from training data, not trained

    def forward(self,features):
        x=features.float() #[batch, 196, 768] fp16 cache -> fp32
        if self.pool_to_49:
            batch=x.shape[0]
            grid=x.view(batch,14,14,x.shape[-1]).permute(0,3,1,2) #[batch, 768, 14, 14], patches are row-major
            grid=F.avg_pool2d(grid,kernel_size=2) #[batch, 768, 7, 7], mean of each 2x2 block
            x=grid.flatten(2).permute(0,2,1) #[batch, 49, 768]
        return self.fc2(self.act(self.fc1(x)))*self.output_scale #Z(I) = g_phi(F(I)) in R^{T x 1024}


class PromptBuilder:
    """Builds token ids for <|im_start|>user\\n[visual tokens]\\n question<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n."""

    def __init__(self,tokenizer,visual_tokens,caps):
        self.tokenizer=tokenizer
        self.visual_tokens=visual_tokens #196, or 49
        self.caps=caps #content-token caps: question 80, caption 64, answer 16
        marker="@@CONTENT@@"
        text=tokenizer.apply_chat_template([{"role":"user","content":marker}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        before,after=text.split(marker)
        self.before_ids=tokenizer(before,add_special_tokens=False)["input_ids"] #chat header up to the user content
        self.after_ids=tokenizer(after,add_special_tokens=False)["input_ids"] #end of user turn + assistant header + empty think block
        self.newline_ids=tokenizer("\n",add_special_tokens=False)["input_ids"]
        self.image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.end_id=tokenizer.convert_tokens_to_ids("<|im_end|>")

    def prompt(self,question,with_image):
        """Prompt ids. No image: the visual tokens and their newline are left out, the text is unchanged."""
        question_ids=self.tokenizer(question,add_special_tokens=False)["input_ids"][:int(self.caps["question"])]
        content=[]
        if with_image==True:
            content=[self.image_token_id]*self.visual_tokens+self.newline_ids #visual tokens before the question
        return self.before_ids+content+question_ids+self.after_ids

    def target(self,text,kind,include_end):
        """Supervised ids: capped caption/answer tokens, plus <|im_end|> when include_end."""
        if kind=="caption":
            cap=int(self.caps["caption"]) #64
        else:
            cap=int(self.caps["answer"]) #16
        ids=self.tokenizer(text,add_special_tokens=False)["input_ids"][:cap]
        if include_end==True:
            ids=ids+[self.end_id]
        return ids


def collate_right(examples,pad_id,device):
    """examples: list of (prompt_ids, target_ids). Right padded ids, labels (-100 outside the target), mask and positions."""
    width=0
    for prompt_ids,target_ids in examples:
        if len(prompt_ids)+len(target_ids)>width:
            width=len(prompt_ids)+len(target_ids)
    ids=[]
    labels=[]
    mask=[]
    for prompt_ids,target_ids in examples:
        sequence=prompt_ids+target_ids
        extra=width-len(sequence)
        ids.append(sequence+[pad_id]*extra)
        labels.append([-100]*len(prompt_ids)+target_ids+[-100]*extra) #visual tokens, prompt and padding are masked
        mask.append([1]*len(sequence)+[0]*extra)
    ids=torch.tensor(ids,device=device)
    labels=torch.tensor(labels,device=device)
    mask=torch.tensor(mask,device=device)
    positions=(mask.cumsum(-1)-1).clamp(min=0) #0,1,2,... over real tokens
    return ids,labels,mask,positions


def embed_inputs(qwen,connector,input_ids,features,image_token_id):
    """Text embeddings with the placeholder positions replaced by connector outputs. features: [n_images, 196, 768] or None."""
    embeds=qwen.get_input_embeddings()(input_ids) #frozen fp16 text embeddings
    if features is not None:
        visual=connector(features) #[n_images, T, 1024], gradients flow back into the connector
        positions=input_ids==image_token_id #[batch, seq], T True values per row with an image
        embeds=embeds.clone()
        embeds[positions]=visual.reshape(-1,visual.shape[-1]).to(embeds.dtype) #row-major order matches the stacked features
    return embeds


def per_example_nll(qwen,embeds,mask,positions,labels):
    """Mean next-token nll over each example's supervised tokens -> [batch]. lm_head only on supervised positions."""
    hidden=qwen.model(inputs_embeds=embeds,attention_mask=mask,position_ids=positions).last_hidden_state
    hidden=hidden[:,:-1,:] #position t predicts t+1
    targets=labels[:,1:]
    valid=targets!=-100
    rows=valid.nonzero()[:,0] #which example each supervised token belongs to
    logits=qwen.lm_head(hidden[valid]).float()
    token_nll=F.cross_entropy(logits,targets[valid],reduction="none") #[n_supervised]
    batch=labels.shape[0]
    total=torch.zeros(batch,device=token_nll.device).index_add(0,rows,token_nll)
    count=torch.zeros(batch,device=token_nll.device).index_add(0,rows,torch.ones_like(token_nll))
    return total/count.clamp(min=1)


def greedy_generate(qwen,connector,prompts,features,image_token_id,pad_id,stop_ids,max_new_tokens):
    """Left-padded batch, greedy decoding with the kv cache. Returns generated token lists (stop token removed)."""
    device=qwen.get_input_embeddings().weight.device
    width=0
    for prompt in prompts:
        if len(prompt)>width:
            width=len(prompt)
    ids=[]
    mask=[]
    for prompt in prompts:
        extra=width-len(prompt)
        ids.append([pad_id]*extra+prompt)
        mask.append([0]*extra+[1]*len(prompt))
    ids=torch.tensor(ids,device=device)
    mask=torch.tensor(mask,device=device)
    positions=(mask.cumsum(-1)-1).clamp(min=0) #left padding does not shift the real positions
    batch=len(prompts)
    generated=[]
    for i in range(batch):
        generated.append([])
    finished=[False]*batch
    with torch.no_grad():
        embeds=embed_inputs(qwen,connector,ids,features,image_token_id)
        output=qwen(inputs_embeds=embeds,attention_mask=mask,position_ids=positions,use_cache=True)
        past=output.past_key_values
        next_tokens=output.logits[:,-1,:].argmax(-1) #greedy
        last_positions=positions[:,-1]
        for step in range(max_new_tokens):
            for i in range(batch):
                if finished[i]==False:
                    token=int(next_tokens[i].item())
                    if token in stop_ids:
                        finished[i]=True #<|im_end|> or eos ends the answer
                    else:
                        generated[i].append(token)
            if all(finished) or step==max_new_tokens-1:
                break
            mask=torch.cat([mask,torch.ones(batch,1,dtype=mask.dtype,device=device)],dim=1)
            last_positions=last_positions+1
            output=qwen(input_ids=next_tokens[:,None],attention_mask=mask,position_ids=last_positions[:,None],past_key_values=past,use_cache=True)
            past=output.past_key_values
            next_tokens=output.logits[:,-1,:].argmax(-1)
    return generated,finished
