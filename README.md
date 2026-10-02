# llm-engine

A from-scratch LLM inference engine for Qwen2.5-0.5B, built to measure where decode time goes and how far each optimization moves it.

Status: Day 1 — forward pass.

## Run
```
pip install torch transformers safetensors
python -m tests.test_logits --tiny   # architecture check, no download
python -m tests.test_logits          # vs Qwen2.5-0.5B-Instruct
```
