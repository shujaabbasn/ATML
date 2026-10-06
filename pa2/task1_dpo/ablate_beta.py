from __future__ import annotations

import argparse
from common.data import load_yaml

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_adapter, load_evaluation_bundle, summary_row
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train",action="store_true") #reuse adapters that already exist (e.g. after a colab disconnect)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])
    budget=int(cfg["short_ablation_examples"]) #600, same first 600 pairs for every beta
    results_dir=repo_path(cfg["results_dir"]) #results/task1_dpo/
    rows=[] #one comparison row per condition

    #the standard one-epoch run goes in the table too, marked as a different budget
    standard_summary_path=results_dir/"standard"/"eval_summary.json" #written by task1_dpo.evaluate --name standard
    if standard_summary_path.exists():
        rows.append(summary_row(load_json(standard_summary_path),"full_epoch")) #not matched to the forks, flagged in the budget column

    for beta in cfg["betas"]:
        beta=float(beta) #yaml gives a float already, this is just to be sure
        run_name="beta_"+str(beta) #beta_0.03, beta_0.1, beta_0.3
        output="outputs/task1_dpo/"+run_name #adapter folder (git ignored)
        if args.skip_train==False:
            #fresh lora on the original policy every time: run_training calls load_policy(fresh_lora=True) and set_seed
            run_training(args.config,run_name,None,output,beta,budget) #only beta changes between forks
            clear_gpu() #free the training model before evaluation
        bundle=load_evaluation_bundle(args.config,output) #same loader for every fork
        summary=evaluate_adapter(bundle,output,run_name,beta) #same prompts, decoding and seed for every fork
        del bundle #drop the models
        clear_gpu() #free memory for the next fork
        rows.append(summary_row(summary,"short_"+str(budget))) #add to the table

    table=pd.DataFrame(rows) #comparison table
    table.to_csv(results_dir/"beta_ablation.csv",index=False) #for the report table/figure
    save_json(results_dir/"beta_ablation.json",rows) #same numbers as json
    print(table.to_string()) #quick look


if __name__ == "__main__":
    main()
