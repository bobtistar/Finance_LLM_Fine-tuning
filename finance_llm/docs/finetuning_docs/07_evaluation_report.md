# 07. Evaluation Report

## 1. Purpose

This document records the full validation evaluation results for the EXAONE classifier, the prompt update that was applied after error analysis, and the final outcome after re-evaluation.

The goals of this stage were:

```text
1. Run full validation on 304 samples to secure reliable metrics
2. Analyze primary_errors.jsonl to find repeated category-boundary failures
3. Strengthen the prompt so inference can separate ambiguous categories more clearly
4. Decide whether the model is good enough for deployment as a classifier node
```

## 2. Evaluation Setup

```text
Base model: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
Adapter: outputs/exaone-3.5-2.4b-finance-qlora/checkpoint-342
Validation set: 304 samples
Inference mode: 4-bit GPU
Output format: JSON only
```

During setup, the following issues had to be resolved:

```text
- EXAONE remote code on Hugging Face main conflicted with local transformers version
- adapter config and peft version mismatch
- adapter path mismatch
- smoke test / eval prompt was too short to express category boundaries well
```

The final stable setup used:

```text
- pinned EXAONE model revision
- fixed adapter path
- 4-bit GPU inference
- updated prompt with stronger category rules
```

## 3. Baseline Result

### Baseline summary.json

```json
{
  "total_count": 304,
  "json_valid_count": 304,
  "invalid_count": 0,
  "json_valid_rate": 1.0,
  "primary_accuracy": 0.7894736842105263,
  "primary_macro_f1": 0.7853915348089203,
  "primary_weighted_f1": 0.7891975620945952,
  "secondary_micro_f1": 0.48148148148148145,
  "secondary_macro_f1": 0.4430848430848431
}
```

### Baseline classification report summary

```text
industry_trend    precision 0.82 / recall 0.80 / f1 0.81
growth_driver     precision 0.77 / recall 0.76 / f1 0.77
earnings_outlook  precision 0.69 / recall 0.95 / f1 0.80
industry_analysis precision 0.54 / recall 0.58 / f1 0.56
company_analysis  precision 0.87 / recall 0.68 / f1 0.76
risk_factor       precision 0.85 / recall 0.85 / f1 0.85
valuation         precision 1.00 / recall 0.91 / f1 0.95
```

### Baseline interpretation

```text
- JSON validity was excellent
- primary classification was usable but still below the target line
- industry_analysis and company_analysis were the main weak points
- earnings_outlook was over-predicted when numbers appeared in the sentence
```

## 4. Error Analysis on Baseline

Baseline primary error count:

```text
64 errors
```

Main confusion pairs:

```text
company_analysis -> earnings_outlook: 10
company_analysis -> growth_driver: 7
growth_driver -> company_analysis: 6
industry_trend -> growth_driver: 5
industry_trend -> earnings_outlook: 4
growth_driver -> industry_trend: 4
company_analysis -> industry_trend: 4
growth_driver -> earnings_outlook: 4
industry_trend -> industry_analysis: 3
```

Main findings:

```text
1. company_analysis was often pulled into earnings_outlook when the sentence contained numbers or estimates
2. company_analysis and growth_driver overlapped when the sentence described capacity, investment, product mix, or operational change
3. industry_analysis and industry_trend overlapped when industry structure and market direction were compressed into one sentence
```

## 5. Prompt Update Applied

The prompt was strengthened to explicitly separate the four categories that caused most errors:

```text
- growth_driver
- earnings_outlook
- industry_analysis
- company_analysis
```

### Main rule additions

```text
1. Numbers alone do not make a sentence earnings_outlook
2. Company internal status, business-unit mix, production lines, customers, and current operations should prefer company_analysis
3. Expansion, partnerships, new products, technical edge, and new market entry should prefer growth_driver when they are framed as future growth logic
4. Market-share comparison, supply chain, value chain, industry structure, and competitor comparison should prefer industry_analysis
5. Industry-wide demand/supply and market direction should prefer industry_trend
```

### Files updated

```text
finance_llm/evaluation/smoke_test_exaone.py
finance_llm/prepare_qlora_dataset.py
finance_llm/labeler.py
```

## 6. Re-evaluation Result After Prompt Update

### Updated summary.json

```json
{
  "total_count": 304,
  "json_valid_count": 304,
  "invalid_count": 0,
  "json_valid_rate": 1.0,
  "primary_accuracy": 0.8453947368421053,
  "primary_macro_f1": 0.8496208801707145,
  "primary_weighted_f1": 0.8456311507946283,
  "secondary_micro_f1": 0.5260273972602739,
  "secondary_macro_f1": 0.5071418042844661
}
```

### Updated classification report summary

```text
industry_trend    precision 0.93 / recall 0.85 / f1 0.89
growth_driver     precision 0.77 / recall 0.78 / f1 0.77
earnings_outlook  precision 0.78 / recall 0.95 / f1 0.85
industry_analysis precision 0.73 / recall 0.67 / f1 0.70
company_analysis  precision 0.87 / recall 0.81 / f1 0.84
risk_factor       precision 0.92 / recall 0.92 / f1 0.92
valuation         precision 1.00 / recall 0.96 / f1 0.98
```

## 7. Before/After Comparison

```text
primary_accuracy:     0.7895 -> 0.8454
primary_macro_f1:     0.7854 -> 0.8496
primary_weighted_f1:  0.7892 -> 0.8456
secondary_micro_f1:   0.4815 -> 0.5260
secondary_macro_f1:   0.4431 -> 0.5071
primary error count:  64 -> 47
```

Most important category improvements:

```text
company_analysis recall: 0.68 -> 0.81
industry_analysis f1:    0.56 -> 0.70
industry_trend f1:       0.81 -> 0.89
valuation f1:            0.95 -> 0.98
```

What this means:

```text
- the prompt update had a real effect
- improvement was not limited to a single class
- the largest baseline weakness, company_analysis, recovered meaningfully
- JSON stability stayed perfect
```

## 8. Remaining Error Pattern After Update

Updated primary error count:

```text
47 errors
```

Top remaining confusion pairs:

```text
growth_driver -> company_analysis: 7
industry_trend -> growth_driver: 7
company_analysis -> earnings_outlook: 6
company_analysis -> growth_driver: 6
growth_driver -> earnings_outlook: 5
industry_analysis -> industry_trend: 2
earnings_outlook -> growth_driver: 2
```

Current bottlenecks are now much narrower:

```text
1. growth_driver <-> company_analysis
2. company_analysis -> earnings_outlook
3. industry_trend -> growth_driver
```

Interpretation:

```text
- the model is no longer broadly unstable
- most remaining errors are concentrated in a few adjacent category boundaries
- future gains will likely come more from relabeling and retraining than from more prompt wording alone
```

## 9. Deployment Decision

Decision:

```text
Deployable as a primary classifier node
```

Reasoning:

```text
- JSON valid rate = 1.0
- primary accuracy = 0.845
- macro F1 = 0.850
- risk_factor and valuation are strong
- company_analysis recovered enough to make routing practical
```

Recommended usage:

```text
- use primary as the routing signal
- use secondary as supporting metadata, not as a hard decision signal
- keep a fallback path only for downstream business logic that is high-impact
```

## 10. Next Recommended Work

For immediate deployment:

```text
1. stop the RunPod after backing up the latest eval results
2. keep the updated prompt version as the current inference baseline
3. wire the classifier into the LangGraph node with primary-first routing
```

For the next improvement cycle:

```text
1. review 20-30 remaining company_analysis / growth_driver / earnings_outlook errors
2. tighten label definitions on ambiguous sentence patterns
3. regenerate training data with the updated instruction
4. run a second QLoRA training cycle if higher accuracy is needed
```
