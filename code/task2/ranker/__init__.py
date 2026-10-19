"""Internals of the FAM-Bench Task 2 (comparative dish analysis) evaluator.

- io_utils    path / string / json helpers
- shuffle     seeded per-question option-letter permutation (position debiasing)
- parsing     strict JSON ranking parser (no regex letter salvage)
- scoring     per-question top-k / MRR / Kendall tau / decision-inversion rate
- knowledge   builds the disease_food_reference block for the KI modes from the
              same knowledge base Task 1 uses (knowledge_base/disease_food_kb.json)
- prompts     system prompts, KI addendum, user payload, output schema
- providers   openai / anthropic / gemini / vllm calls with image attachment
"""
