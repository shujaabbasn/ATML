from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel

from common.utils import load_json, load_yaml, peak_vram_gib, repo_path, save_json, wall_timer


#runs the frozen siglip encoder once over every image we use and stores the 196 patch features in fp16
#siglip's last_hidden_state already has the final post_layernorm applied (exactly once), we do not apply it again
#this script is the only place siglip is loaded, so it is never on the gpu while qwen trains


def all_images(splits):
    images=set()
    for row in splits["caption_train"]+splits["caption_dev"]:
        images.add(row["image"])
    for name in ["grounding_dev","grounding_test","vqa_train","vqa_dev","vqa_test"]:
        for pair in splits[name]:
            for example in pair["examples"]:
                images.add(example["image"])
    for image in splits["grounding_pools"]["images"]:
        images.add(image)
    return sorted(images)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--batch-size",type=int,default=64)
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    device="cuda"
    splits=load_json(cfg["data"]["splits"])
    images=all_images(splits)
    image_dir=repo_path(cfg["data"]["image_dir"])

    processor=SiglipImageProcessor.from_pretrained(cfg["vision_encoder"]) #resize to 224, rescale, normalize, no random crop
    encoder=SiglipVisionModel.from_pretrained(cfg["vision_encoder"],torch_dtype=torch.float32).to(device)
    encoder.eval()
    torch.cuda.reset_peak_memory_stats()
    timer=wall_timer()
    features=torch.empty(len(images),int(cfg["model"]["visual_tokens"]),int(cfg["model"]["vision_dim"]),dtype=torch.float16) #[N, 196, 768]
    for start in range(0,len(images),args.batch_size):
        batch_keys=images[start:start+args.batch_size]
        pil_images=[]
        for key in batch_keys:
            pil_images.append(Image.open(image_dir/key).convert("RGB"))
        pixel_values=processor(images=pil_images,return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            output=encoder(pixel_values=pixel_values)
        patch_features=output.last_hidden_state #[batch, 196, 768], 14x14 patches in row-major spatial order, post layernorm
        features[start:start+len(batch_keys)]=patch_features.to(torch.float16).cpu()
        if start%(args.batch_size*20)==0:
            print(start,"/",len(images))
    seconds=timer()
    index={}
    for i in range(len(images)):
        index[images[i]]=i
    torch.save({"index":index,"features":features},str(repo_path(cfg["data"]["features"])))
    info={"images":len(images),"shape":list(features.shape),"dtype":"float16","seconds":seconds,"peak_vram_gib":peak_vram_gib(),
        "encoder":cfg["vision_encoder"],"feature_bytes":features.numel()*2,"post_layernorm_applied_once":True}
    save_json(repo_path(cfg["results_dir"])/"feature_cache.json",info)
    print(info)


if __name__=="__main__":
    main()
