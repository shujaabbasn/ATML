"""Runs the staff `setup_data.py --prefix-banks` unchanged, with one compatibility shim.

The staff script calls model.generate(..., generator=gen). transformers 4.57.1 (pinned by the manual) has no
`generator` argument and raises "model_kwargs are not used by the model: ['generator']".
The shim removes that argument and seeds the global torch RNG with the same seed instead
(gen.initial_seed() = the staff formula 6304 + 10007*j (+123 for the draft)), so every prompt still gets
its own fixed seed and the bank is reproducible. Nothing else in the staff script is touched.
Run from the pa4 folder: python kaggle_runs/staff_prefix_banks.py
"""
import sys
import torch
from transformers.generation.utils import GenerationMixin

original_generate=GenerationMixin.generate #keep the real one


def generate_with_seed(self,*args,**kwargs):
    generator=kwargs.pop("generator",None) #4.57.1 does not accept it
    if generator is not None:
        torch.manual_seed(generator.initial_seed()) #same seed the staff generator carried, seeds cpu and every gpu
    return original_generate(self,*args,**kwargs)


def main():
    GenerationMixin.generate=generate_with_seed #Qwen2ForCausalLM inherits generate from here
    sys.path.insert(0,"data_repo/scripts")
    import setup_data
    sys.argv=["setup_data.py","--prefix-banks"]
    setup_data.main()


if __name__=="__main__":
    main()
