from __future__ import annotations

import argparse
import collections
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common.utils import load_yaml, repo_path, save_json
from task2_vlm.vqa_eval import normalize_answer


#builds data/task2/splits.json and downloads only the coco images we use
#  coco captions          6,000 train images x 3 captions, 96 val images x 1 caption (dev)
#  annotation grounding   reusable presence/counting pools over the 6,000 caption train images,
#                         24 dev pairs and 64 test pairs from val2014 (half presence, half counting)
#  natural vqav2          complementary pairs, consensus >= 8/10 on both images and different answers
#dev and test images are val2014, disjoint from each other and from training (train2014)

PLURALS={"person":"people","mouse":"mice","sheep":"sheep","skis":"skis","scissors":"scissors","knife":"knives","broccoli":"broccoli",
    "bus":"buses","bench":"benches","sandwich":"sandwiches","couch":"couches","toothbrush":"toothbrushes","wine glass":"wine glasses"} #irregular coco names, the rest just get an "s"


def plural(name):
    if name in PLURALS:
        return PLURALS[name]
    else:
        return name+"s"


def presence_question(name):
    return "Does this image contain "+name+"?" #manual template, the category name is used as written in coco


def counting_question(name):
    return "How many "+plural(name)+" can you see?" #manual template


def download(url,dest):
    dest=repo_path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        print("downloading",url)
        urllib.request.urlretrieve(url,dest)
    return dest


def read_json_from_zip(zip_path,file_name):
    """Load one json file from a zip, found by its file name so a folder prefix inside the zip does not matter."""
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.split("/")[-1]==file_name:
                with z.open(member) as f:
                    return json.load(f)
    raise FileNotFoundError(file_name+" not in "+str(zip_path))


def image_key(split,image_id):
    """split/file name, the same string is used as the feature cache key."""
    return split+"/COCO_"+split+"_"+str(image_id).zfill(12)+".jpg"


def consensus(answers):
    """Most common normalized answer, how many of the 10 annotators gave it, and one original string from that group."""
    normalized=[]
    for a in answers:
        normalized.append(normalize_answer(a["answer"]))
    counts=collections.Counter(normalized)
    best,count=counts.most_common(1)[0]
    original=None
    for a in answers:
        if normalize_answer(a["answer"])==best:
            original=a["answer"] #first original string of the consensus group = training target
            break
    return best,count,original


def eligible_vqa_pairs(questions,annotations,pairs,split,min_agreement):
    """Complementary pairs whose two consensus answers differ and have >= 8/10 agreement each."""
    question_by_id={}
    for q in questions["questions"]:
        question_by_id[q["question_id"]]=q
    annotation_by_id={}
    for a in annotations["annotations"]:
        annotation_by_id[a["question_id"]]=a
    eligible={"yes/no":[],"number":[],"other":[]}
    for q1,q2 in pairs:
        if q1 not in annotation_by_id or q2 not in annotation_by_id:
            continue
        a1=annotation_by_id[q1]
        a2=annotation_by_id[q2]
        if question_by_id[q1]["question"]!=question_by_id[q2]["question"] or a1["answer_type"]!=a2["answer_type"]:
            continue #same question, same type, otherwise it is not a clean pair
        c1,n1,t1=consensus(a1["answers"])
        c2,n2,t2=consensus(a2["answers"])
        if n1<min_agreement or n2<min_agreement or c1==c2:
            continue
        examples=[]
        for qid,annotation,canonical,target in [(q1,a1,c1,t1),(q2,a2,c2,t2)]:
            references=[]
            for a in annotation["answers"]:
                references.append(a["answer"]) #all ten kept for soft accuracy
            examples.append({"question_id":qid,"image":image_key(split,annotation["image_id"]),"target":target,"canonical":canonical,"references":references})
        eligible[a1["answer_type"]].append({"pair_id":"vqa_"+str(q1)+"_"+str(q2),"suite":"vqa","qtype":a1["answer_type"],"question":question_by_id[q1]["question"],"examples":examples})
    for qtype in eligible:
        eligible[qtype].sort(key=pair_sort_key) #fixed order before the seeded draw
    return eligible


def pair_sort_key(pair):
    return pair["pair_id"]


def annotation_sort_key(annotation):
    return annotation["id"]


def count_type(pairs,qtype):
    total=0
    for pair in pairs:
        if pair["qtype"]==qtype:
            total=total+1
    return total


def take_disjoint(candidates,count,rng,used_images):
    """Walk the candidates in a seeded random order and keep pairs whose images are unused."""
    chosen=[]
    for index in rng.permutation(len(candidates)):
        pair=candidates[index]
        images=[pair["examples"][0]["image"],pair["examples"][1]["image"]]
        if images[0] in used_images or images[1] in used_images or images[0]==images[1]:
            continue
        chosen.append(pair)
        used_images.add(images[0])
        used_images.add(images[1])
        if len(chosen)==count:
            break
    if len(chosen)<count:
        raise RuntimeError("not enough disjoint pairs")
    return chosen


def object_counts(instances,split,allowed_images):
    """image -> category name -> number of non-crowd boxes, -1 when the category has a crowd box (ambiguous count)."""
    category_name={}
    for c in instances["categories"]:
        category_name[c["id"]]=c["name"]
    counts={}
    for a in instances["annotations"]:
        key=image_key(split,a["image_id"])
        if allowed_images is not None and key not in allowed_images:
            continue
        name=category_name[a["category_id"]]
        if key not in counts:
            counts[key]={}
        if a["iscrowd"]==1:
            counts[key][name]=-1 #crowd box: present, but no reliable count
        elif counts[key].get(name,0)>=0:
            counts[key][name]=counts[key].get(name,0)+1
    return counts,sorted(category_name.values())


def make_presence_pair(rng,name,positives,negatives,pair_id):
    positive=positives[rng.integers(len(positives))]
    negative=negatives[rng.integers(len(negatives))]
    question=presence_question(name)
    return {"pair_id":pair_id,"suite":"grounding","qtype":"presence","category":name,"question":question,"examples":[
        {"image":positive,"target":"yes","canonical":"yes","references":["yes"]},
        {"image":negative,"target":"no","canonical":"no","references":["no"]}]}


def make_counting_pair(rng,name,by_count,pair_id):
    available=sorted(by_count.keys()) #counts between 1 and 4 that have images
    picked=rng.choice(available,size=2,replace=False) #two different counts
    examples=[]
    for count in picked:
        images=by_count[int(count)]
        image=images[rng.integers(len(images))]
        examples.append({"image":image,"target":str(int(count)),"canonical":str(int(count)),"references":[str(int(count))]})
    return {"pair_id":pair_id,"suite":"grounding","qtype":"counting","category":name,"question":counting_question(name),"examples":examples}


def grounding_eval_pairs(counts,categories,image_pool,n_presence,n_counting,rng,used_images,prefix,count_min,count_max):
    """Fixed dev/test grounding pairs from val images, no image used twice anywhere in dev/test."""
    pairs=[]
    while count_type(pairs,"presence")<n_presence:
        name=categories[rng.integers(len(categories))]
        positives=[]
        negatives=[]
        for image in image_pool:
            if image in used_images:
                continue
            if name in counts.get(image,{}):
                positives.append(image)
            else:
                negatives.append(image)
        if len(positives)==0 or len(negatives)==0:
            continue
        pair=make_presence_pair(rng,name,positives,negatives,prefix+"_presence_"+str(len(pairs)))
        for example in pair["examples"]:
            used_images.add(example["image"])
        pairs.append(pair)
    while count_type(pairs,"counting")<n_counting:
        name=categories[rng.integers(len(categories))]
        by_count={}
        for image in image_pool:
            if image in used_images:
                continue
            count=counts.get(image,{}).get(name,0)
            if count>=count_min and count<=count_max:
                if count not in by_count:
                    by_count[count]=[]
                by_count[count].append(image)
        if len(by_count)<2:
            continue
        pair=make_counting_pair(rng,name,by_count,prefix+"_counting_"+str(len(pairs)))
        for example in pair["examples"]:
            used_images.add(example["image"])
        pairs.append(pair)
    return pairs


def add_mismatch_mapping(pairs,rng):
    """Manual: fix the mismatched image before generation, from another pair in the same suite, never from the original pair."""
    for i in range(len(pairs)):
        for example in pairs[i]["examples"]:
            other=i
            while other==i:
                other=int(rng.integers(len(pairs)))
            side=int(rng.integers(2))
            example["mismatch_image"]=pairs[other]["examples"][side]["image"]
            example["mismatch_pair_id"]=pairs[other]["pair_id"]


def download_images(cfg,images):
    image_dir=repo_path(cfg["data"]["image_dir"])
    base_url=cfg["urls"]["coco_images"]
    def fetch(key):
        dest=image_dir/key
        if dest.exists():
            return 0
        dest.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve(base_url+"/"+key,dest)
        return 1
    with ThreadPoolExecutor(max_workers=16) as pool:
        fetched=sum(pool.map(fetch,sorted(images)))
    print("downloaded",fetched,"new images,",len(images),"needed")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    args=ap.parse_args()
    cfg=load_yaml(args.config)
    data_cfg=cfg["data"]
    raw=cfg["data"]["raw_dir"]
    rng=np.random.default_rng(int(cfg["seed"])) #one seeded generator for every selection, used in a fixed order

    coco_zip=download(cfg["urls"]["coco_annotations"],raw+"/annotations_trainval2014.zip")
    vqa_files={}
    for name in ["vqa_questions_train","vqa_questions_val","vqa_annotations_train","vqa_annotations_val","vqa_pairs_train","vqa_pairs_val"]:
        vqa_files[name]=download(cfg["urls"][name],raw+"/"+name+".zip")

    #natural vqav2: train pairs
    eligible_train=eligible_vqa_pairs(
        read_json_from_zip(vqa_files["vqa_questions_train"],"v2_OpenEnded_mscoco_train2014_questions.json"),
        read_json_from_zip(vqa_files["vqa_annotations_train"],"v2_mscoco_train2014_annotations.json"),
        read_json_from_zip(vqa_files["vqa_pairs_train"],"v2_mscoco_train2014_complementary_pairs.json"),
        "train2014",int(data_cfg["vqa_min_agreement"]))
    vqa_train=[]
    for qtype,count in data_cfg["vqa_train_pairs"].items():
        indices=rng.choice(len(eligible_train[qtype]),size=int(count),replace=False)
        for index in sorted(indices):
            vqa_train.append(eligible_train[qtype][index])

    #natural vqav2: dev and test pairs from val2014, image disjoint
    eligible_val=eligible_vqa_pairs(
        read_json_from_zip(vqa_files["vqa_questions_val"],"v2_OpenEnded_mscoco_val2014_questions.json"),
        read_json_from_zip(vqa_files["vqa_annotations_val"],"v2_mscoco_val2014_annotations.json"),
        read_json_from_zip(vqa_files["vqa_pairs_val"],"v2_mscoco_val2014_complementary_pairs.json"),
        "val2014",int(data_cfg["vqa_min_agreement"]))
    used_val=set() #every val image already given to dev or test
    vqa_dev=[]
    vqa_test=[]
    for qtype in ["yes/no","number","other"]:
        vqa_dev=vqa_dev+take_disjoint(eligible_val[qtype],int(data_cfg["vqa_dev_pairs"][qtype]),rng,used_val)
    for qtype in ["yes/no","number","other"]:
        vqa_test=vqa_test+take_disjoint(eligible_val[qtype],int(data_cfg["vqa_test_pairs"][qtype]),rng,used_val)

    #coco captions: 6,000 train images x 3 captions
    captions_train=read_json_from_zip(coco_zip,"captions_train2014.json")
    by_image={}
    for a in sorted(captions_train["annotations"],key=annotation_sort_key):
        key=image_key("train2014",a["image_id"])
        if key not in by_image:
            by_image[key]=[]
        by_image[key].append(a["caption"].strip())
    candidates=[] #train images with at least 3 captions
    for key in sorted(by_image):
        if len(by_image[key])>=3:
            candidates.append(key)
    chosen=rng.choice(len(candidates),size=int(data_cfg["caption_train_images"]),replace=False)
    caption_train=[]
    for index in sorted(chosen):
        key=candidates[index]
        picks=rng.choice(len(by_image[key]),size=int(data_cfg["captions_per_train_image"]),replace=False)
        for pick in sorted(picks):
            caption_train.append({"image":key,"caption":by_image[key][pick]})
    del captions_train
    caption_images=set()
    for row in caption_train:
        caption_images.add(row["image"])
    caption_images=sorted(caption_images)

    #annotation-derived grounding pools over the caption train images
    instances_train=read_json_from_zip(coco_zip,"instances_train2014.json")
    train_counts,categories=object_counts(instances_train,"train2014",set(caption_images))
    del instances_train
    pools={"images":caption_images,"categories":categories,"counts":train_counts} #sampled from during training

    #grounding dev/test from val images not used by the vqa pairs
    instances_val=read_json_from_zip(coco_zip,"instances_val2014.json")
    val_counts,_=object_counts(instances_val,"val2014",None)
    del instances_val
    val_pool=sorted(val_counts.keys()) #val images with at least one annotation, also used as negatives
    count_min=int(data_cfg["count_min"])
    count_max=int(data_cfg["count_max"])
    grounding_dev=grounding_eval_pairs(val_counts,categories,val_pool,int(data_cfg["grounding_dev_pairs"])//2,int(data_cfg["grounding_dev_pairs"])//2,rng,used_val,"dev",count_min,count_max)
    grounding_test=grounding_eval_pairs(val_counts,categories,val_pool,int(data_cfg["grounding_test_pairs"])//2,int(data_cfg["grounding_test_pairs"])//2,rng,used_val,"test",count_min,count_max)

    #caption dev: 96 unused val images, one caption each
    captions_val=read_json_from_zip(coco_zip,"captions_val2014.json")
    val_captions={}
    for a in sorted(captions_val["annotations"],key=annotation_sort_key):
        key=image_key("val2014",a["image_id"])
        if key not in val_captions:
            val_captions[key]=[]
        val_captions[key].append(a["caption"].strip())
    caption_dev=[]
    val_keys=sorted(val_captions.keys())
    for index in rng.permutation(len(val_keys)):
        key=val_keys[index]
        if key in used_val:
            continue
        pick=int(rng.integers(len(val_captions[key])))
        caption_dev.append({"image":key,"caption":val_captions[key][pick]})
        used_val.add(key)
        if len(caption_dev)==int(data_cfg["caption_dev_images"]):
            break

    add_mismatch_mapping(grounding_test,rng)
    add_mismatch_mapping(vqa_test,rng)

    #split isolation check (manual step 1)
    train_images=set(caption_images)
    for pair in vqa_train:
        for example in pair["examples"]:
            train_images.add(example["image"])
    dev_images=set()
    caption_dev_images=[]
    for row in caption_dev:
        dev_images.add(row["image"])
        caption_dev_images.append(row["image"])
    test_images=set()
    for pair in grounding_dev+vqa_dev:
        for example in pair["examples"]:
            dev_images.add(example["image"])
    for pair in grounding_test+vqa_test:
        for example in pair["examples"]:
            test_images.add(example["image"])
    isolation={"train_images":len(train_images),"dev_images":len(dev_images),"test_images":len(test_images),
        "train_dev_overlap":len(train_images&dev_images),"train_test_overlap":len(train_images&test_images),"dev_test_overlap":len(dev_images&test_images)}
    assert isolation["train_dev_overlap"]==0 and isolation["train_test_overlap"]==0 and isolation["dev_test_overlap"]==0

    splits={"seed":int(cfg["seed"]),"caption_train":caption_train,"caption_dev":caption_dev,"grounding_pools":pools,
        "grounding_dev":grounding_dev,"grounding_test":grounding_test,"vqa_train":vqa_train,"vqa_dev":vqa_dev,"vqa_test":vqa_test}
    save_json(data_cfg["splits"],splits)

    #ids only, small enough for the repo (manual: save the selected image, question and pair ids)
    ids={"caption_train_images":caption_images,"caption_dev_images":sorted(caption_dev_images)}
    for name in ["grounding_dev","grounding_test","vqa_train","vqa_dev","vqa_test"]:
        ids[name]=[]
        for pair in splits[name]:
            entry={"pair_id":pair["pair_id"],"qtype":pair["qtype"],"question":pair["question"],"images":[],"question_ids":[],"targets":[],"mismatch_images":[]}
            for example in pair["examples"]:
                entry["images"].append(example["image"])
                entry["question_ids"].append(example.get("question_id")) #only natural vqa has question ids
                entry["targets"].append(example["target"])
                entry["mismatch_images"].append(example.get("mismatch_image")) #only test pairs have one
            ids[name].append(entry)
    eligible_train_counts={}
    eligible_val_counts={}
    for qtype in eligible_train:
        eligible_train_counts[qtype]=len(eligible_train[qtype])
        eligible_val_counts[qtype]=len(eligible_val[qtype])
    counts={"eligible_vqa_train":eligible_train_counts,"eligible_vqa_val":eligible_val_counts,
        "caption_train_examples":len(caption_train),"caption_dev":len(caption_dev),"grounding_dev_pairs":len(grounding_dev),"grounding_test_pairs":len(grounding_test),
        "vqa_train_pairs":len(vqa_train),"vqa_dev_pairs":len(vqa_dev),"vqa_test_pairs":len(vqa_test),"isolation":isolation}
    save_json(repo_path(cfg["results_dir"])/"selected_ids.json",ids)
    save_json(repo_path(cfg["results_dir"])/"data_summary.json",counts)
    print(counts)

    download_images(cfg,train_images|dev_images|test_images)


if __name__=="__main__":
    main()
