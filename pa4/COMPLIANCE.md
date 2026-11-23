# PA4 manual → code

Each row pairs a manual requirement (paraphrased from the full PA4 manual text) with the code that meets it and the result file it writes. The last section lists the places where the manual leaves a choice open and what was chosen.

## Shared conventions

| Manual requirement | Where |
|---|---|
| transformers 4.57.1, huggingface_hub 0.36.2 | `requirements.txt` |
| Seed 6304, extra fixed seeds where specified | every `configs/*.yaml`; `common/utils.set_seed` |
| Record checkpoint, tokenizer, versions, GPU, CUDA | `package_versions()` saved in `setup.json`, `attention_results.json`, `c1_reference_check.json`, `cost.json`, `throughput.json` |
| Warmed-up, CUDA-synchronized timing; separate prefill from decode | `common/utils.time_call` (CUDA events); `task1_compression/evaluate.resources` (decode minus prompt prefill) |
| Use the staff data, pinned revisions, no silent substitutions | `data_repo/` + `local_source_revisions.json`; every loader goes through `data_path` |
| Record every reported hyperparameter | `config` / `config_used.json` saved with each task's results |
| Do not change conditions after seeing outcomes; keep failures (OOM, skipped) | OOM / skipped rows are kept in `attention_results.json`; every P3 λ candidate kept in `selection_dev.json` |

## Task 1

| Manual requirement | Where |
|---|---|
| Qwen2.5-1.5B base, fp16, each condition from an untouched copy | `model_utils.load_model`, `backup_state` / `restore_state` called before every condition in `run.py` |
| Eligible = q,k,v,o,gate,up,down in every block; embeddings, norms, biases, LM head uncompressed; tied weights counted once | `model_utils.eligible_linears`, `fp16_bytes_outside` (deduplicates by data pointer) |
| 128×512 C4 calibration, same IDs for Wanda/AWQ/sensitivity | `calibration.block_stats` is the only reader of calibration data |
| 32-sequence dev set for selection only | `run.dev_windows`, used only by `p3` λ selection |
| PPL on the exact 50k scored tokens of WikiText-2 and C4 val; first token excluded | `evaluate.perplexity` checks `scored_tokens` against every staff record |
| HellaSwag 2000 examples, sum of log-likelihood, acc and character-normalized acc; first 512 = diagnostic; template recorded | `evaluate.hellaswag`, `HELLASWAG_TEMPLATE` in `setup.json`; `hellaswag_diagnostic512` in metrics |
| Sparsity, nominal bits, packed storage incl. scales/zero-points, prefill 512/2048, 64-token decode, peak memory, ≥5 warmed reps | `run.Runner.storage`, `evaluate.resources`, `configs/task1.yaml: timing` (repeats 5) |
| Unstructured sparse format = 1 mask bit + 16 bits per nonzero | `model_utils.sparse_bytes_matrix` |
| Fake quantization labelled as such; no FLOP/VRAM/latency claims from zeros or nominal bits | `storage.in_memory_representation`; `t1_resources` table shows in-memory vs packed |
| P1 magnitude, per-output-row, s ∈ {0.25, 0.5, 0.75}; layer-output error on calibration activations | `pruning.magnitude_prune`; `layer_relative_error_calibration` in every `metrics.json` |
| P2 Wanda, same groups and data; compare score distributions | `pruning.wanda_prune` (+ `score_distributions.json`) |
| P3 sensitivity measure + allocation rule, 0.10 ≤ s_l ≤ 0.85, matched to P1/P2 by integer counting, at 50% and 75% | `pruning.owl_outlier_ratio`, `owl_allocation`, `exact_row_counts`; `t1_p3_budget_s*` table |
| Q1 asymmetric group-128 RTN at 8/4/3 bits; formulas and padding stated | `quantization.fake_quantize` (docstring = formulas) |
| Q2 AWQ-style: fold inverse scales or keep fp16 scales and count them; verify the rescaling identity; calibration reconstruction error; say what of AWQ is reproduced | `quantization.awq_quantize`, `apply_group_scale`, `check_scaling_identity`; `q2_scaling_identity.json`; `t1_awq_reconstruction` |
| Q3 widths {2,3,4,8}, same quantizer, packed bytes matching Q2 INT4 and INT3, exact assignment reported | `quantization.greedy_bit_allocation`; `q3_mixed_match_int*/allocation.json` |
| 2:4 diagnostic at 50% Wanda, no T4 acceleration claimed | `pruning.nm_prune`, `nm24_wanda_s50` note |
| Quality vs packed bytes, pruning and quantization separated | `make_report_assets.py → t1_quality_vs_bytes.pdf` |
| Sensitivity visualization per adaptive method | `t1_p3_sensitivity_allocation.pdf`, `t1_q3_sensitivity_bits.pdf` |
| Paired 95% bootstrap for P3−P2 and Q3−Q2: HellaSwag examples; perplexity by whole windows, pooled token NLL before exp | `common.utils.bootstrap_ppl_diff` (window lengths from `token_nll.npz`), `compare.py → paired_bootstrap.json` |
| Uniform vs adaptive layer-error diagnostic on dev data | `layer_relative_error_dev` in metrics → `t1_layer_error_uniform_vs_adaptive.pdf` |
| Fixed qualitative HellaSwag example across conditions | `t1_qualitative_hellaswag` (selection rule written in the table comment) |
| Test sets never used for P3/Q3 choices | P3 uses dev perplexity; Q3 uses calibration-sample error only |

## Task 2

| Manual requirement | Where |
|---|---|
| A1 explicit QKᵀ, causal mask, stable softmax; causality by perturbing the future; FP32 reference for outputs and gradients; edge cases | `exact_attention.dense_attention`, `causality_check`, `compare_to_reference`, `edge_cases` |
| A2 exact blockwise online softmax, verified across tile sizes and dtypes; not labelled FlashAttention | `exact_attention.tiled_attention` (fp32 m, l, acc), `tile_sweep` (tiles 32–256) |
| A3 SDPA on identical tensors; B=2, H=4, d=64, L 256–8192, fp16/fp32; 10 warmed reps; forward and forward+backward; peak allocated and reserved memory; OOM recorded; backend recorded | `benchmark` (`*_peak_mb`, `*_peak_reserved_mb`), `sdpa_backends` + `default_backend` |
| MQAR via staff generator, invariants checked first, frozen test/dev | `mqar.main` (`validate` on every record), `training_stream(K, regime)` |
| 2 pre-norm blocks, d=128, 4 heads of 32, 4× MLP, learned token embeddings, sinusoidal positions, vocab 256; same init/budget for every operator | `MQARModel`, `sinusoidal`, `build_model` (hybrid gate added after the shared init) |
| AdamW 8e-4, wd 0.01, batch 32, 4,500 updates, 200 warmup, linear decay to 15%, clip 1.0, eval at 500…4500, seeds 6304–6306 | `configs/task2.yaml: mqar`, `lr_lambda`, `train_one` |
| Query-position classification, loss only at query positions; chance 1/64 and 1/K | `labels_for`, `evaluate`; `chance` in `summary.json` |
| Test: 1,024 sequences per (K, regime); accuracy per regime and per distance bin [0,32), [32,64), [64,128), [128,256) | `accuracy_table`, `DISTANCE_BINS` |
| W=64 two-layer reach limit 2(W−1)=126 reported as architectural | `reach_matrix`, `max_reach`, `reachable_fraction` |
| B1: dense K=4/8/16 mixed + K=16 near; window K=4/8/16 near + K=16 mixed; all tested on near/far/mixed; dense with W=64 mask at test time only | `configs/task2.yaml: mqar.runs`, `mask_control`, `set_masks` → `t2b_mask_control` |
| B2: ELU+1 linear K=4/8/16 mixed; local/dilated K=4 and 16 (2 heads W=64, 2 heads offsets 0,4,…,252); pairs, state, runtime, candidate keys; masked dense not called a sparse kernel | `allowed_mask`, `linear_attention`, `attention_pairs`, `state_floats`, `candidate_keys`, `note` in `result.json` |
| Forward-only sweep T ∈ {256,1024,4096,8192}, random embeddings, matched B/H/d; mask / prefix-tensor materialization stated | `mqar --inference_sweep → inference_sweep.json → t2b_inference_sweep` |
| B3 at K=16, 3 seeds, 4,500 updates, within 5% parameters; prediction stated before evaluation; state bytes per layer matched to the linear comparator with a documented precision adjustment; selection rule | `hybrid` (W=16, fp16 state); `check_b3_state`; `b3_prediction`, `b3_rule` in the config (saved before training); 5% check raises in `train_one` → `t2b_b3_intervention` |
| C Qwen2.5-1.5B-Instruct fp16, staff 180 cases, measured token lengths | `kv_cache.build_case` (`encoded_length`, `context_length`) |
| C1 full cache vs no cache, greedy agreement, finite logits | `step_c1` |
| C2 sink+recent and question-aware eviction, largest legal FP16 count within the INT4 budget (indices counted); 8 sinks + 64 recent in both; retention per layer and kv head, stated | `compress_eviction`, `eviction_keep`, `question_scores` |
| C3 token-wise vs channel-K/token-V INT4, group 64, fp16 scale+zero, fp16 tail, same rule for later tokens; K and V reconstruction errors separately; teacher-forced divergence | `compress_quant` (`k_relative_error`, `v_relative_error`), `residual_quantize`, `packed_int4_bytes`, `kl_rows` |
| C4 hybrid: sinks + recent + top question-aware in FP16, the rest channel-K/token-V INT2 in complete 64-element groups, residual FP16; metadata, indices counted; not above the budget | `hybrid_plan`, `compress_hybrid` |
| Shared question/decode FP16 component reported separately; realized fraction of full FP16 bytes per prompt | `shared_decode_fp16_bytes`, `total_live_cache_bytes`, `fraction_of_full_fp16_bytes` in `c_eval_rows.jsonl` |
| C5 five caches + reference on 180 cases: EM, reference-answer NLL, KL, evidence retained, bytes; ≥2,000 paired bootstrap resamples; accuracy conditional on reference correctness; breakdown by length / position / distractors | `step_eval`, `step_summary` (`accuracy_given_reference_correct`, `accuracy_by_factor`) |
| C5 packed encode/decode at batch 1 and 8, both lengths, ≥10 reps; decode step with unpacking vs FP16 at matched shapes; serialized vs allocated | `packed_benchmark` → `c5_timing.json: packed` → `t2c_packed_codec` |
| Evidence offsets only for evaluation | used only in `evidence_kept` |

## Task 3

| Manual requirement | Where |
|---|---|
| 1.5B-Instruct target/teacher, 0.5B-Instruct draft/student, same tokenizer + chat template + EOS | `common.load_tokenizer` (equality check), `chat_ids`, `eos_ids` |
| Step 1: draft, verify, accept/reject, residual, fallback, rollback, bonus, variable γ, EOS | `sampler.verify_block`, `speculative_generate` |
| 100k-trial Markov χ² at α=0.01 with expected ≥10, and a rejected negative control | `sampler_tests.toy_tests`, `chi_square` |
| Cache rollback / bonus / prefill-equivalence / target-only checks; variable proposal lengths, final-token truncation, EOS stop, one-token (γ=1) and fully accepted rounds; padding n/a at batch 1 | `sampler_tests.real_tests` (`cap5_ok`, `eos_stop_ok`, `gamma1_ok`, `variable_proposal_lengths_seen`, `draft_equals_target_acceptance`) |
| Step 2: pretrained draft, γ ∈ {2,4}; accepted/examined, accepted/proposed, accepted prefix per verification, target passes/token, draft forwards, memory, EOS, paired time; prefill and decode separately; isolated draft/target step cost at several prompt lengths | `spec_eval --mode speed --drafts original`, `--mode throughput` (`latency`); `banks --bank exact_spec`, `--bank consistency` |
| Training data = 500 GSM8K + 300 short; no test prompts | `common.train_prompts`; `overlap_check.py` |
| 40k scored tokens (30k math + 10k short), caps 192/32, T=1, top-p=1 | `configs/task3.yaml: distill` |
| One update per 256 scored tokens, short last update, microbatch 1 | `distill.Trainer` |
| AdamW 3e-4, wd 0.01, fp16 frozen backbone, fp32 adapters/optimizer, fp16 autocast + GradScaler, unscale before clipping, clip 1.0, KL normalized per update | `Trainer.__init__`, `Trainer.add`, `Trainer.step` |
| Per presentation: source id, rollout seed, response ids, termination reason, teacher-scored tokens | `presentations.jsonl` |
| Loss / LR / teacher-scored-token curves; teacher generation, teacher scoring, student rollout, backward updates and merging timed separately | `updates.jsonl`, `cost.json`, `merge_check.json: merge_seconds` → `t3_distill_cost`, `t3_distill_curves` |
| LoRA r=8, α=16, dropout 0, q/k/v/o; seeds 6304, 6305; same init for off/on | `distill.train` |
| Skipped/overflowed updates and quota truncation recorded | `updates.jsonl` (`skipped_overflow`), `cost.json` (`truncations`) |
| Forward KL over the full vocabulary, sampled token never a hard label | `distill.kl_positions` |
| Off-policy uses fixed teacher responses, generation cost included; on-policy uses fresh student rollouts | `teacher_bank`, `train` (`presentations.jsonl`, `cost.json`) |
| Step 5 forward/reverse KL, TV, 1−TV, top-1 on the same frozen prefixes, incl. each student's own bank and the other student's; split by math / familiar / shifted and position bins; seed dispersion; aligned disagreements | `distribution.py` (`summary.json`, `seed_dispersion.json`, `aligned_disagreements.jsonl`), `banks --bank student` |
| Step 6 merge, check merged vs unmerged; same EOS-aware 120-prompt evaluation at γ=2,4 with a 64-token cap; counterbalanced adjacent target-only run per condition; four-part per-round time + prefill + host; c, v_γ, Ŝ, measured S, break-even α; one unmerged adapter timed | `distill.merge`, `spec_eval --mode speed / throughput / unmerged` |
| Fixed-64-token diagnostic on 24 math prompts, EOS ignored for timing only | `spec_eval --mode throughput` → `t3_speedup` |
| Paired bootstrap 95% intervals over prompts for throughput and acceptance | `spec_eval.summary` (`throughput_ratio_vs_target`, `speed_differences`) → `t3_speed`, `t3_speed_bootstrap` |
| Step 7 by math / familiar / shifted / output-length bins; standalone student accuracy at 512 / 32 with extraction and truncation flags; deployment break-even incl. distillation + merge cost | `spec_eval --mode quality`, `spec_eval.summary` (`quality`, `deployment`) → `t3_quality`, `t3_deployment` |
| GSM8K exact numeric match; short-form exact match | `common.is_correct` |

## Before you submit

| Manual item | Where |
|---|---|
| Calibration/training data not overlapping held-out evaluations | `overlap_check.py → results/overlap_check.json` (all MQAR training streams) |
| Adaptive allocations/hyperparameters chosen without test results | P3: dev only; Q3: calibration only; B3 window fixed a priori from state matching |
| Compressed models compared at the stated sparsity/byte budgets | `t1_p3_budget_s*`, `t1_q3_budget` |
| Speculative sampler passed its distribution and cache tests | `sampler_tests.json` |
| Timings correspond to the implementation described | notes in `t1_resources`, `t2b_costs`, `t2c_systems` |
| Every number traces to a result file | `make_report_assets.py` builds every table and figure from `results/` |
| Repo: code, configs, environment, data scripts, example IDs, per-example results, README, attribution; no raw data or big checkpoints | `README.md`, `.gitignore`; per-example files: `hellaswag_scores.jsonl`, `token_nll.npz`, `test_queries.npz`, `c_eval_rows.jsonl`, `speed/*.jsonl`, `quality/*.jsonl`, `distribution/per_prefix.jsonl.gz` |

## Choices the manual leaves open

1. Dilated (K=4 and 16) and hybrid (K=16) runs train on the mixed regime. The manual gives their K but not the regime; mixed matches their linear comparator.
2. B3 hybrid window W=16 with fp16 buffer and state, fixed in advance by the state-byte match with the fp32 ELU+1 linear model (16,640 vs 16,896 bytes per layer). No dev search. The linear state is stored as a running mean, not a running sum. The sum overflowed fp16 in the first three hybrid runs (NaN), so only those were rerun after the fix (logs kept in `results/task2/hybrid_fp16_sum_overflow/`).
3. Question-aware scores are summed over question tokens and the GQA query heads, and kept separately per layer and per kv head.
4. KV position indices for per-head FP16 retention are counted as int32 (4 bytes).
5. Isolated decode-cost prefix lengths 128 / 512 / 1024; the unmerged adapter timed is `off_policy_s6304`.
