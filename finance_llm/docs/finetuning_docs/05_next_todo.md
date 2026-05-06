# 05. Next TODO List

## A. Evaluation Status

Completed:

```text
[x] Full validation evaluation on valid.jsonl (304 samples)
[x] summary.json review
[x] classification_report.txt review
[x] confusion pattern review through primary_errors.jsonl
[x] prompt update applied and re-evaluation completed
```

Current best evaluation result:

```text
JSON valid rate      1.000
Primary accuracy     0.845
Primary macro F1     0.850
Primary weighted F1  0.846
Secondary micro F1   0.526
Secondary macro F1   0.507
```

## B. Deployment TODO

```text
[ ] Freeze the current inference prompt version
[ ] Keep the latest eval_results as deployment baseline evidence
[ ] Connect EXAONE classifier to LangGraph classifier node
[ ] Route downstream logic by primary category
[ ] Use secondary only as supporting metadata
[ ] Decide whether high-impact paths need fallback handling
```

## C. Code Cleanup TODO

```text
[ ] Extract EXAONE inference code into a reusable class/module
[ ] Move shared category definitions and prompt text into one source
[ ] Separate parser / prompt / model-loading logic more cleanly
[ ] Normalize adapter path configuration
```

Suggested structure:

```text
src/
  models/
    exaone_classifier.py
  prompts/
    category_prompt.py
  utils/
    json_parser.py
  evaluation/
    eval_exaone_classifier.py
```

## D. Remaining Model Improvement TODO

Main remaining confusion pairs:

```text
growth_driver -> company_analysis
industry_trend -> growth_driver
company_analysis -> earnings_outlook
company_analysis -> growth_driver
growth_driver -> earnings_outlook
```

Recommended next improvement tasks:

```text
[ ] Review 20-30 remaining primary_errors samples
[ ] Tighten label boundaries for company_analysis / growth_driver / earnings_outlook
[ ] Add more industry_analysis examples if that class remains weak in production
[ ] Regenerate dataset with the updated instruction if retraining is planned
[ ] Run second-round QLoRA only if deployment feedback shows the need
```

## E. Retraining Ideas

Current best baseline before retraining:

```text
Prompt-updated inference already reached primary accuracy 0.845
```

If another training round is needed, try only one or two changes at a time:

### Experiment v2: lower learning rate

```text
lr = 1e-4
epoch = 2
r = 16
alpha = 32
dropout = 0.05
```

### Experiment v3: higher adapter capacity

```text
lr = 1e-4 or 2e-4
epoch = 2
r = 32
alpha = 64
dropout = 0.05
```

### Experiment v4: more training

```text
lr = 1e-4
epoch = 3
r = 16
alpha = 32
dropout = 0.05
```

Guideline:

```text
Do not change too many hyperparameters at once.
Compare every new run against the current deployment baseline:
primary accuracy 0.845 / macro F1 0.850 / JSON valid rate 1.0
```
