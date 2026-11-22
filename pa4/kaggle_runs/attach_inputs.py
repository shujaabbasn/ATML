"""Copies earlier Kaggle sessions' outputs into this checkout.

Attach earlier notebooks' outputs with Add Input -> Notebook output. Kaggle mounts them somewhere under /kaggle/input
(the exact folder layout differs between Kaggle versions), so this walks the whole tree and merges:
  - every .../pa4/data_repo/data folder into data_repo/data (prepared C4/WikiText/HellaSwag/GSM8K files, prefix banks, revision lock)
  - every .../pa4/results folder into results/
  - any data_repo.zip / *_results.zip, in case only the zips were kept
Run from the pa4 folder after cloning the staff repo into data_repo.
"""
import os
import shutil
import zipfile

INPUT="/kaggle/input"


def main():
    if os.path.exists("data_repo")==False:
        raise RuntimeError("clone the staff repo into data_repo first")
    copied=0
    zips=[]
    data_folders=[]
    result_folders=[]
    for root,dirs,files in os.walk(INPUT):
        if ".git" in dirs:
            dirs.remove(".git") #never walk into a git folder
        for name in files:
            if name=="data_repo.zip" or name.endswith("_results.zip"):
                zips.append(os.path.join(root,name))
        parent=os.path.basename(root)
        if parent=="data_repo" and "data" in dirs:
            data_folders.append(os.path.join(root,"data"))
        if parent=="pa4" and "results" in dirs:
            result_folders.append(os.path.join(root,"results"))

    #zips first, so the unzipped folders (same content, maybe newer) win
    for path in sorted(zips):
        print("unzip",path)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(".") #paths inside are data_repo/... or results/...
        copied=copied+1
    for path in sorted(data_folders):
        print("copy",path,"-> data_repo/data")
        shutil.copytree(path,"data_repo/data",dirs_exist_ok=True)
        copied=copied+1
    for path in sorted(result_folders):
        print("copy",path,"-> results")
        shutil.copytree(path,"results",dirs_exist_ok=True)
        copied=copied+1

    if copied==0:
        print("nothing attached, data will be built from scratch")
    else:
        print("merged",copied,"inputs")
    if os.path.exists("data_repo/data/local_source_revisions.json"):
        print("revision lock present") #setup_data reuses these commits instead of resolving new ones
    else:
        print("no revision lock yet")


if __name__=="__main__":
    main()
