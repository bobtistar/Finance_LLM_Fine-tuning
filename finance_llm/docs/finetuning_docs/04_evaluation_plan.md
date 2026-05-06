# 04. Evaluation Plan

## 1. Objective

The goal of this evaluation is not to measure a fully general-purpose model. The goal is to decide whether the EXAONE classifier is stable and accurate enough to be attached to a LangGraph classifier node.

The expected behavior is:

```text
sentence / paragraph input
-> primary category classification
-> up to 2 secondary categories
-> strict JSON output
```

So the focus is on both:

```text
1. classification quality
2. output stability
```

## 2. Evaluation Data

```text
valid.jsonl
```

Validation set size:

```text
304 samples
```

## 3. Metrics

```text
1. JSON valid rate
2. Primary accuracy
3. Primary macro F1
4. Primary weighted F1
5. Per-category precision / recall / F1
6. Secondary micro F1
7. Secondary macro F1
8. Confusion matrix
9. Invalid output sample review
10. Primary error sample review
```

## 4. Deployment Criteria

Direct deployment as classifier node:

```text
JSON valid rate >= 0.98
primary accuracy around 0.85 or higher
macro F1 >= 0.75
critical categories such as risk_factor and valuation should not collapse
```

Deployable with caution or fallback:

```text
JSON valid rate >= 0.95
primary accuracy around 0.80 ~ 0.85
macro F1 around 0.65 ~ 0.75
```

Retraining recommended:

```text
JSON valid rate < 0.95
primary accuracy < 0.80
macro F1 < 0.65
```

## 5. Categories That Need Extra Attention

Important categories:

```text
industry_analysis
risk_factor
valuation
```

Likely confusion zones:

```text
industry_trend <-> industry_analysis
growth_driver <-> company_analysis
earnings_outlook <-> valuation
earnings_outlook <-> company_analysis
```

## 6. Evaluation Output Files

The evaluation script should generate:

```text
eval_results/
  summary.json
  classification_report.txt
  confusion_matrix.csv
  invalid_samples.jsonl
  primary_errors.jsonl
```

## 7. How to Read the Result

Check these first in summary.json:

```text
json_valid_rate
primary_accuracy
primary_macro_f1
primary_weighted_f1
secondary_micro_f1
secondary_macro_f1
invalid_count
```

Interpretation:

```text
high JSON valid rate
-> low parser risk in LangGraph

high primary accuracy
-> primary category is usable for routing

high macro F1
-> minority classes are not collapsing

low secondary F1
-> secondary should be treated as supporting metadata only
```

For primary_errors.jsonl:

```text
Review at least 20~30 samples manually
Classify each error into:
A. model truly wrong
B. gold label ambiguous
C. category definition unclear
```

If B or C appears often, prompt and label definition work may be more valuable than raw retraining.

## 8. Actual Outcome

### Baseline result

```text
json_valid_rate = 1.0
primary_accuracy = 0.789
primary_macro_f1 = 0.785
secondary_micro_f1 = 0.481
secondary_macro_f1 = 0.443
```

### After prompt update

```text
json_valid_rate = 1.0
primary_accuracy = 0.845
primary_macro_f1 = 0.850
primary_weighted_f1 = 0.846
secondary_micro_f1 = 0.526
secondary_macro_f1 = 0.507
```

### Final decision

```text
Deployable as a primary classifier node
Use primary as routing signal
Use secondary as supporting metadata
Future improvement should focus on
company_analysis / growth_driver / earnings_outlook boundaries
```
