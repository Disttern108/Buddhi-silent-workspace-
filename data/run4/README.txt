Fine-tune (single combined run, per protocol):
  mlx_lm.lora --model Qwen/Qwen3-0.6B-Base --train \
    --data <this dir with train_combined.jsonl renamed train.jsonl> \
    --iters 800 --learning-rate 1e-5 --batch-size 4

mlx-lm 'completions' format is used: {prompt, completion}.
Evaluate with test_recall.jsonl / test_transfer.jsonl by
generating from each prompt and string-matching the answer
token (exact match on invented names is a fair metric).

A-facts vs B-facts are DISJOINT entity sets: compare recall
and transfer accuracy on A-entities vs B-entities.
