# PA4: Compression, Efficient Attention and Speculative Decoding

Code for CS 6304 PA4. Three tasks, all run on a Kaggle T4 (fp16, no bf16).

| Task | Folder | What it does |
|---|---|---|
| 1 | `task1_compression/` | Qwen2.5-1.5B pruning (P1 magnitude, P2 Wanda, P3 OWL-style nonuniform, 2:4 diagnostic) and weight-only quantization (Q1 RTN, Q2 AWQ-style, Q3 mixed precision) |
| 2 | `task2_attention/` | A: dense / tiled online-softmax / SDPA attention. B: MQAR with dense, local, linear, dilated and a gated hybrid. C: KV-cache eviction, INT4 layouts and a hybrid cache on Qwen2.5-1.5B-Instruct |
| 3 | `task3_speculative/` | Exact speculative sampling, off-policy and on-policy forward-KL distillation of Qwen2.5-0.5B-Instruct towards Qwen2.5-1.5B-Instruct, distribution metrics and speedup analysis |

Configs are in `configs/`, results are written to `results/task*/`. `common/utils.py` holds IO, timing and bootstrap helpers.

## Data

All data comes from the staff repository [atml-pa4-datasets](https://github.com/AbDu11aHHH/atml-pa4-datasets), cloned into `pa4/data_repo/` (not committed):

```bash
git clone https://github.com/AbDu11aHHH/atml-pa4-datasets.git data_repo
cd data_repo && pip install -r requirements-public.txt
python scripts/setup_data.py --public          # C4, WikiText-2, HellaSwag, GSM8K windows
python scripts/setup_data.py --prefix-banks    # teacher + original-draft prefix banks (GPU)
python scripts/check_setup.py
```

Keep `data_repo/data/local_source_revisions.json`. It pins the Hugging Face commits that the report has to cite. Set `PA4_DATA` to use a data folder somewhere else.

## Running

Every command runs from `pa4/`. The notebooks in `kaggle_runs/` only launch these commands, one Kaggle session each. Each notebook clones this repo and the staff data repo, then `kaggle_runs/attach_inputs.py` merges the prepared data and the results of any earlier notebook attached as an input. Speed and quality runs write `<condition>.jsonl.part` and rename it only when every prompt is done, so a session cut off by the time limit can be rerun and redoes only the unfinished condition. The staff prefix banks are built by `kaggle_runs/staff_prefix_banks.py`, which runs the staff `setup_data.py --prefix-banks` unchanged except that the `generator=` argument it passes to `model.generate` (rejected by transformers 4.57.1) is replaced by seeding torch with that generator's seed, so each prompt keeps the staff seed 6304 + 10007·j (+123 for the draft).

```bash
# Task 1
python -m task1_compression.run --condition baseline   # also p1 p2 p3 nm24 q1 q2 q3, or all
python -m task1_compression.compare                    # paired bootstrap P3-P2, Q3-Q2 and summary.csv

# Task 2
python -m task2_attention.exact_attention
python -m task2_attention.mqar --arch all             # the 14 B1/B2/B3 runs in configs/task2.yaml x 3 seeds
python -m task2_attention.mqar --inference_sweep       # forward-only sweep, T = 256..8192
python -m task2_attention.kv_cache --step c1           # then eval, timing, summary

# Task 3
python -m task3_speculative.sampler_tests
python -m task3_speculative.banks --bank exact_spec
python -m task3_speculative.banks --bank consistency
python -m task3_speculative.distill --step teacher_bank
python -m task3_speculative.distill --step train --method off_policy --seed 6304   # x2 methods x2 seeds
python -m task3_speculative.banks --bank student --name off_policy_s6304           # each trained draft
python -m task3_speculative.distribution
python -m task3_speculative.spec_eval --mode speed --drafts original off_policy_s6304 ...       # 64-token EOS-aware benchmark
python -m task3_speculative.spec_eval --mode quality --drafts original off_policy_s6304 ...     # standalone answers, caps 512 / 32
python -m task3_speculative.spec_eval --mode throughput --drafts original off_policy_s6304 ...  # fixed 64 tokens, 24 math prompts, cost model
python -m task3_speculative.spec_eval --mode unmerged                                          # one adapter timed unmerged
python -m task3_speculative.spec_eval --mode summary
```

## Design decisions (what the report has to state)

### Task 1
- Eligible weights are q, k, v, o, gate, up and down in all 28 blocks. Embeddings, norms, biases and the tied LM head stay fp16 and are counted once.
- Every condition restores the untouched fp16 state dict first.
- Perplexity scores exactly the staff `score_target_offsets`, so the first token of each window is never a target. Per-token NLL and the window lengths are saved; the paired bootstrap resamples whole windows and pools their token NLL before exponentiating.
- Every condition also saves the per-block output error on calibration sequences (and on development sequences for the allocation diagnostic).
- HellaSwag uses the lm-eval cleaning and template, and the score is the sum of the ending's log-probs. acc_norm divides by the ending's character length. The first 512 examples are reported separately as the diagnostic subset.
- Pruning masks work per output row. Wanda runs block by block: the norms for block l are measured after blocks 0..l-1 are already pruned, as in the original code.
- P3 sensitivity is the OWL outlier ratio (share of Wanda scores above 5x the matrix mean), with one transformer block as the unit. Sparsity is mapped linearly onto [S-λ, S+λ], clipped to [0.10, 0.85] and shifted back to the target. Per-row counts are then rounded so the global count is exactly round(S·N). λ is chosen on the 32 development sequences only, and every candidate, including λ=0 (uniform), is kept in `selection_dev.json`.
- Quantizer, shared by Q1/Q2/Q3: asymmetric, group 128 along input columns, s=(max-min)/(2^b-1), z=round(-min/s). Storage per group is the packed codes plus an fp16 scale and a b-bit zero-point. Weights are fake-quantized, so the matmuls stay dense fp16.
- Q2 follows the AWQ scale search s=mean|x|^α, normalized, with α on a 21-point grid. 1/s is folded into input_layernorm (q,k,v), post_attention_layernorm (gate,up) and up_proj rows (down). o_proj cannot be folded under GQA (v_proj has 256 rows against o_proj's 1536 columns), so its fp16 scale is kept and counted. Not reproduced: weight-clipping search, and the loss is per-linear output error rather than whole-module output error. `q2_scaling_identity.json` is the check that scaling without quantization leaves the outputs unchanged.
- Q3 sensitivity is each projection's relative output error at 2/3/4/8 bits on the calibration sample. Allocation is greedy: the upgrade with the largest error drop per extra byte goes first, under the packed-byte budget of Q2 INT4 or INT3. The leftover gap is saved.
- Storage formats: unstructured sparse is 1 mask bit + 16 bits per nonzero (as the manual specifies). 2:4 is fp16 values + 2-bit indices (our choice). No sparse or low-bit kernel runs, so latency and memory stay close to fp16.

### Task 2
- A: the tiled version is plain PyTorch with autograd. It is not a FlashAttention kernel. Its output and gradients are checked for tile sizes 32-256 in both dtypes. SDPA backends are forced one at a time to record which ones run, and the default one is read from the profiler. Peak allocated and reserved memory are both saved.
- B: every operator shares the same embeddings, fixed sinusoidal positions, MLP, norms and initialization (same seed; the hybrid gate is added afterwards). AdamW at peak LR 8e-4, 200 warmup updates, linear decay to 15%, 4,500 updates of 32 sequences, clip 1.0, seeds 6304-6306. Each run trains one architecture at one K and one regime, exactly the manual's list: dense K=4/8/16 mixed + K=16 near; local W=64 K=4/8/16 near + K=16 mixed; linear K=4/8/16 mixed; local/dilated K=4 and 16; hybrid K=16. Every (K, regime) has its own fixed training stream, so seeds only change the initialization. The loss and accuracy cover the K query positions only, with argmax over all 256 ids. Intermediate checkpoints are evaluated on dev, and test is evaluated once at the end, reported per (K, regime) and per distance bin [0,32), [32,64), [64,128), [128,256). The dense K=16 mixed model is also tested with the W=64 mask imposed at test time only. Local and dilated attention use a masked dense implementation, so their runtimes are not sparse-kernel runtimes; the forward-only sweep records that. `reach_matrix` gives the architectural reach (126 for two W=64 layers).
- B3 intervention: each head mixes a W=16 local softmax with an ELU+1 linear memory through a sigmoid gate (+0.2% parameters). The comparator is the K=16 ELU+1 linear model. To match its state bytes per layer, the hybrid keeps its local buffer and linear state in fp16 (the manual allows a precision adjustment): 2·(2·32·16 + 32·32 + 32)·4 heads = 16,640 B against 4·(32·32 + 32)·4 = 16,896 B. The window, the fp16 storage and the prediction are fixed in the config before training; there is no dev search. The linear part of the state is stored as the running mean S_i/i, z_i/i: the output is the ratio φ(q)S/φ(q)z, so this is the same function with the same bytes. The first hybrid runs stored the running sum, which grew past fp16's 65,504 in layer 2 (inf, then NaN at updates 1,900 to 3,350 in all three seeds). Their logs are kept in `results/task2/hybrid_fp16_sum_overflow/`, and the reported hybrid comes from `kaggle_runs/task2_hybrid_rerun.ipynb`.
- C: the prompt is split at `QUESTION_BEGIN`. The context cache is compressed at the pre-question point, and the question is then re-run on the compressed cache, so no question KV ever saw evicted tokens. The budget is the manual's M_q(L_ctx). Retention is selected independently per layer and kv head, so each kept FP16 token costs 4·d_h bytes plus a 4-byte position index; eviction keeps floor(B / (n_l H_kv (4 d_h + 4))) tokens per kv head, with 8 sinks + 64 recent in both rules. Question-aware scores are softmax attention from the RoPE'd question queries, summed over question tokens and over the GQA query heads, per layer and kv head. INT4 groups are 64 with fp16 scale and zero. Complete 64-token blocks are quantized and the remainder stays fp16; later tokens follow the same rule. K and V reconstruction errors are saved separately. The hybrid keeps every context token: the 8 sinks, 64 recent and the top question-aware tokens in FP16 (with indices), and the lowest-ranked tokens in complete 64-token channel-wise-K/token-wise-V INT2 groups. The number of INT2 groups is the smallest that fits the INT4 budget, which gives the largest legal FP16 count. The question suffix and generated tokens are a shared FP16 component reported separately. Packed encode/decode and the decode step with unpacking are timed at batch 1 and 8 for both context lengths. Evidence offsets are used only for evaluation.

### Task 3
- Both tokenizers are checked to be identical, and both models use the staff bank's chat template.
- `sampler.verify_block` contains the whole accept/reject/residual/bonus rule, and the toy χ² test runs the same function. Cache rollback is `DynamicCache.crop`.
- Distillation: LoRA r=8, α=16 on q/k/v/o, fp32 adapters, fp16 frozen backbone, GradScaler (skipped updates are logged). Every optimizer update holds exactly 256 teacher-scored tokens, and the 40,000-token quota leaves a final update of 64. A response that crosses an update boundary is re-forwarded with the new weights for its remaining positions. Off-policy runs are charged the teacher-bank generation time.
- Every presentation (source id, rollout seed, response ids, termination, teacher-scored tokens) is logged in `presentations.jsonl`. Teacher scoring, student forward/backward and merging are timed separately.
- Held-out evaluation: the primary speed benchmark uses a 64-token cap with EOS, γ = 2 and 4, two paired seeds per prompt, and an adjacent target-only run on the same prompt and seed whose order alternates (counterbalanced). Answer quality is a separate run of the target and each standalone student at 512 (math) / 32 (short) tokens, with extractable and truncated flags. The fixed-64-token diagnostic ignores EOS on the first 24 GSM8K test prompts.
- Every speculative round is timed as draft proposal, target verification, accept/residual sampling and cache rollback; prefill and host overhead are kept apart. α comes from full rounds, c from isolated single-step latencies at 128/512/1024-token prefixes, v_γ from a (γ+1)-token verification forward; Ŝ = (1 + Σ α^j)/(γc + v_γ) and the break-even α follow the manual's formula.

## Overlap check, report assets, compliance

```bash
python overlap_check.py           # manual "Before You Submit": no calibration/training overlap with held-out data
python make_report_assets.py      # every table (outputs/tables, csv + tex) and figure (outputs/figures) from results/
```

`COMPLIANCE.md` maps each manual requirement to the code and result file that covers it.

## Attribution for reused external code and recipes

- `task3_speculative/banks.bank_positions` copies the position-sampling rule from the staff `scripts/setup_data.py`, so our banks match the staff banks.
- `task1_compression/evaluate.hellaswag_preprocess` follows the HellaSwag cleaning in EleutherAI lm-evaluation-harness.
- The AWQ scale form s = mean|x|^α / sqrt(max·min) follows the AutoAWQ implementation (Lin et al., 2024).
- The OWL allocation (outlier ratio mapped linearly to [S−λ, S+λ]) follows Yin et al. (2024).
- Wanda scores follow Sun et al. (2024), the speculative sampling rule follows Leviathan et al. (2023), and ELU+1 linear attention follows Katharopoulos et al. (2020).
- MQAR data comes from the staff `scripts/mqar_generator.py`.

## Choices the manual leaves open
- Dilated (K=4, 16) and hybrid runs train on the mixed regime, the same as their linear comparator.
- B3 uses W=16 with fp16 state storage, fixed by the state-byte match above.
- KV position indices are counted as int32.
