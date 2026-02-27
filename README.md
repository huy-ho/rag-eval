# rag-eval

A local, production-ready prototype for evaluating RAG (Retrieval-Augmented Generation) pipelines using [DeepEval](https://github.com/confident-ai/deepeval).

Runs entirely on your machine — no API keys, no cloud services. Uses [Ollama](https://ollama.com) as the judge LLM.

---

## What it does

Given a set of Q&A records from your RAG pipeline (question → retrieved contexts → answer), this framework scores each response across 8 metrics and writes a full report per run.

### Metrics

| Metric | Type | Description | Threshold |
|---|---|---|---|
| `faithfulness` | Standard | Does the answer stick to the retrieved context? | 0.65 |
| `answer_relevancy` | Standard | Does the answer address the question? | 0.60 |
| `contextual_precision` | Standard | Are retrieved chunks relevant to the question? | 0.55 |
| `contextual_recall` | Standard | Did chunks cover everything needed to answer? | 0.55 |
| `contextual_relevancy` | Standard | Are the chunks relevant at all? | 0.50 |
| `hallucination` | Standard | Rate of invented facts not in context (**lower = better**) | 0.0 |
| `supply_chain_specificity` | G-Eval | Does the answer include actionable identifiers (SKUs, specs, lead times)? | 0.55 |
| `answer_completeness` | G-Eval | Does the answer cover all key details from the context? | 0.55 |

All thresholds are configurable in `config.yaml`.

---

## Project structure

```
rag-eval/
├── evaluate.py       # Main entrypoint — run this
├── data_loader.py    # Dataset loading (mock / CSV / JSON / JSONL)
├── mock_data.py      # Built-in 10-record Cisco supply chain dataset
├── config.yaml       # All tuneable settings
├── requirements.txt
└── results/          # Auto-created; one timestamped folder per run
    └── 20260227_143022/
        ├── results.xlsx   # Full results + failures (two sheets)
        ├── results.json   # Same data, machine-readable
        ├── run_info.json  # Run metadata (model, thresholds, pass rate, duration)
        └── eval.log       # DEBUG-level log for the run
```

---

## Setup

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and pull the judge model:

```bash
ollama pull mistral
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Also install `pyyaml` if not pulled in transitively:
> ```bash
> pip install pyyaml
> ```

### 3. Run

```bash
python evaluate.py
```

Results are written to `results/<YYYYMMDD_HHMMSS>/`. Re-running never overwrites previous results.

---

## Configuration (`config.yaml`)

```yaml
model: mistral          # Ollama model to use as judge

thresholds:
  faithfulness: 0.65
  answer_relevancy: 0.60
  contextual_precision: 0.55
  contextual_recall: 0.55
  contextual_relevancy: 0.50
  hallucination: 0.0    # zero-tolerance for invented facts
  supply_chain_specificity: 0.55
  answer_completeness: 0.55

output_dir: results     # where run folders are created

max_cases: null         # null = run all; set an integer to cap (e.g. 3)
data_path: null         # null = use built-in mock dataset
```

To run a strict experiment, copy the file and swap the name in the script:

```bash
cp config.yaml config_strict.yaml
# edit config_strict.yaml, then change the load_config() call in evaluate.py
```

---

## Using your own dataset

Set `data_path` in `config.yaml` to a file path. Three formats are supported:

### CSV

```
data_path: my_data.csv
```

Required columns: `question`, `answer`, `contexts`, `ground_truth`.
The `contexts` column is pipe-separated (`chunk 1 | chunk 2 | chunk 3`).

```csv
question,answer,contexts,ground_truth
What is X?,X is a thing.,Context A | Context B,X is a thing with details.
```

### JSON

```
data_path: my_data.json
```

Top-level list of objects:

```json
[
  {
    "question": "What is X?",
    "answer": "X is a thing.",
    "contexts": ["Context A", "Context B"],
    "ground_truth": "X is a thing with details."
  }
]
```

### JSONL

```
data_path: my_data.jsonl
```

One JSON object per line (same fields as above).

---

## Output files

### `run_info.json`

```json
{
  "run_id": "20260227_143022",
  "model": "mistral",
  "dataset": "mock",
  "n_cases": 10,
  "thresholds": { "faithfulness": 0.65, "..." : "..." },
  "pass_rate": 0.8,
  "duration_seconds": 42.1
}
```

### `results.json`

One record per test case with all metric scores, reasons, and pass/fail.

### `results.xlsx`

- **All Results** sheet — one row per test case, all metric scores and LLM reasons
- **Failures** sheet — one row per metric failure, with the reason for the failure

---

## How it works

```
config.yaml
    ↓
evaluate.py           loads config, sets up logging & run folder
    ↓
data_loader.py        loads + validates dataset records
    ↓
build_test_cases()    assembles DeepEval LLMTestCase objects
    ↓
run_evaluation()      calls DeepEval with 8 metrics via Ollama
    │                 (retries up to 3× with exponential backoff)
    ↓
build_results_df()    flattens results into a DataFrame
    ↓
print_summary()       per-question table + metric stats + failure diagnosis
    ↓
results/{run_id}/     xlsx + json + run_info.json + eval.log
```

---

## Extending

- **Add a metric** — instantiate it in `_build_metrics()` in `evaluate.py`, add its key to `THRESHOLDS`, `METRIC_COLS`, and `METRIC_SHORT`.
- **Change the judge model** — set `model: llama3` (or any pulled Ollama model) in `config.yaml`.
- **Stricter runs** — lower thresholds or set `hallucination: 0.0` (already the default).
- **Batch experiments** — keep multiple named configs (e.g. `config_strict.yaml`) and swap the filename in `_load_config()`.
