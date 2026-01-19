# Persona Datasets Documentation

This document describes the structure and usage of the persona conversation datasets generated from the [persona-subspace](https://github.com/lu-christina/persona-subspace) project.

## Overview

We've generated **1,299 HuggingFace datasets** (14GB total) containing conversations where models roleplay specific personas or exhibit specific traits. Each dataset represents a single (model, persona/trait) pair.

### Dataset Counts

- **832 role datasets**: 277 roles × 3 models
- **467 trait datasets**: 240 traits × ~2 models (some filtered)

### Models

- **gemma-2-27b**: 513 datasets (277 roles + 236 traits)
- **llama-3.3-70b**: 277 datasets (277 roles only)
- **qwen-3-32b**: 509 datasets (278 roles + 231 traits)

## Dataset Schema

Each dataset contains conversations in HuggingFace `Dataset` format with the following fields:

### Common Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | Model name (gemma-2-27b, llama-3.3-70b, qwen-3-32b) |
| `system_prompt` | str | Persona instruction given to the model |
| `question` | str | One of 240 shared questions asked across all personas |
| `messages` | list[dict] | Conversation in chat format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` |
| `prompt_index` | int | Which variant of the system prompt (0-4) |
| `question_index` | int | Which of the 240 questions (0-239) |
| `label` | str | Prompting condition (see below) |
| `extract_score` | int | GPT-4-mini quality rating (see below) |

### Role-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `role` | str | Role name (e.g., "accountant", "activist", "alien") |
| `label` | str | `"pos"` (positive) or `"default"` (baseline/no persona) |
| `extract_score` | int | **0-3 scale** rating roleplay quality:<br>• **0**: Refuses to roleplay, responds as AI only<br>• **1**: Generic AI assistant style (polite, bulleted)<br>• **2**: Shows role attributes but identifies as AI<br>• **3**: Fully playing the role ✨ |

### Trait-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `trait` | str | Trait name (e.g., "analytical", "empathetic", "blunt") |
| `label` | str | `"pos"` (positive) or `"neg"` (negative/opposite trait) |
| `extract_score` | int | **0-100 scale** rating trait strength:<br>• **0**: Trait completely absent<br>• **50**: Moderate presence<br>• **70+**: Strong presence ⭐<br>• **100**: Very strongly present |

## Dataset Naming Convention

Datasets follow the pattern: `{model}__{type}__{name}[__{filters}]`

**Examples:**
- `gemma-2-27b__role__accountant` - All accountant conversations for Gemma
- `qwen-3-32b__trait__analytical__min70__pos` - High-quality positive analytical trait for Qwen
- `llama-3.3-70b__role__activist__min3` - Fully in-character activist conversations for Llama

**Filters:**
- `__min{score}`: Minimum extract_score (e.g., `min3` = score ≥ 3, `min70` = score ≥ 70)
- `__pos` or `__neg`: Trait polarity (positive or negative prompting)

## Storage Locations

### Datasets
- **HF format**: `/workspace/datasets/processed/persona/{name}/`
- **Parquet format**: `/workspace/datasets/processed/persona/{name}.parquet`
- **Total size**: 14GB

### Token Analysis
- **Statistics JSON**: `/workspace/persona_token_stats.json` (141KB)
- **Histograms (plots)**: `/workspace/persona_token_plots/{name}.png` (1,299 images)

## Token Statistics

Each dataset has been analyzed for token counts using model-specific tokenizers with chat templates applied:

```json
{
  "metadata": {
    "total_datasets": 1299,
    "timestamp": "2025-10-01T...",
    "git_sha": "..."
  },
  "datasets": {
    "gemma-2-27b__role__accountant": {
      "model": "gemma-2-27b",
      "type": "role",
      "name": "accountant",
      "plot": "/workspace/persona_token_plots/gemma-2-27b__role__accountant.png"
    },
    ...
  }
}
```

## Usage Examples

### Loading a Dataset

```python
from datasets import load_from_disk

# Load a specific role dataset
ds = load_from_disk("/workspace/datasets/processed/persona/gemma-2-27b__role__accountant")

# Access conversations
for row in ds["train"]:
    print(f"Question: {row['question']}")
    print(f"Score: {row['extract_score']}")
    print(f"Conversation: {row['messages']}")
```

### Creating Custom Datasets

```bash
# List available roles for Gemma
uv run python - <<'PY'
from chatspace.persona import list_available_roles
print("\n".join(list_available_roles("gemma-2-27b")))
PY

# Materialise a high-quality role dataset (score == 3) to HF + parquet
uv run python - <<'PY'
from pathlib import Path
from chatspace.persona import load_persona_dataset, save_persona_dataset

dataset = load_persona_dataset(
    model="gemma-2-27b",
    dataset_type="role",
    name="accountant",
    min_score=3,
)
save_persona_dataset(dataset, Path("/workspace/datasets/processed/persona"), "gemma-2-27b__role__accountant__min3")
PY

# Materialise a strong positive trait dataset (score ≥ 70)
uv run python - <<'PY'
from pathlib import Path
from chatspace.persona import load_persona_dataset, save_persona_dataset

dataset = load_persona_dataset(
    model="qwen-3-32b",
    dataset_type="trait",
    name="analytical",
    min_score=70,
    label_filter="pos",
)
save_persona_dataset(dataset, Path("/workspace/datasets/processed/persona"), "qwen-3-32b__trait__analytical__min70__pos")
PY
```

### Batch Generation

### Token Analysis

```bash
# Analyze token counts for all datasets
uv run python scripts/analyze_persona_token_counts.py

# Results saved to:
# - /workspace/persona_token_stats.json (statistics)
# - /workspace/persona_token_plots/*.png (histograms)
```

## Available Personas

### Roles (277 total)

Examples: accountant, activist, actor, alien, analyst, anarchist, architect, artist, blogger, chef, comedian, detective, diplomat, doctor, economist, engineer, explorer, gamer, historian, hacker, inventor, journalist, lawyer, mentor, musician, negotiator, oracle, philosopher, pilot, poet, rebel, scientist, teacher, visionary, warrior, writer, and many more...

**Full list**: Run the snippet above or call `chatspace.persona.list_available_roles("gemma-2-27b")`

### Traits (240 total)

Examples: absolutist, abstract, accessible, analytical, anxious, arrogant, artistic, assertive, benevolent, blunt, calculating, calm, casual, cautious, chaotic, compassionate, confident, cooperative, creative, critical, curious, cynical, decisive, diplomatic, direct, eccentric, empathetic, enthusiastic, ethical, extroverted, flexible, formal, friendly, gentle, honest, idealistic, impulsive, independent, innovative, intense, intuitive, logical, methodical, optimistic, patient, pragmatic, precise, rebellious, reserved, ruthless, scholarly, skeptical, strategic, tactful, thoughtful, traditional, transparent, unconventional, verbose, whimsical, and many more...

**Full list**: Run the snippet above or call `chatspace.persona.list_available_traits("gemma-2-27b")`

## Use Cases

### 1. Fine-Tuning
Train models to exhibit specific personalities or traits:
```python
# Load high-quality trait data
ds = load_from_disk("/workspace/datasets/processed/persona/qwen-3-32b__trait__analytical__min70__pos")

# Use for LoRA training, DPO, or standard fine-tuning
```

### 2. Steering Vectors
Create trait steering vectors from pos/neg pairs:
```python
# Load positive and negative trait datasets
pos_ds = load_from_disk("qwen-3-32b__trait__analytical__pos")
neg_ds = load_from_disk("qwen-3-32b__trait__analytical__neg")

# Compute steering vector: mean(pos) - mean(neg)
```

### 3. Analysis
Study how models respond to different personas:
```python
# Analyze token distributions by score
import json
with open("/workspace/persona_token_stats.json") as f:
    stats = json.load(f)

# Compare across models
gemma_accountant = stats["datasets"]["gemma-2-27b__role__accountant"]
llama_accountant = stats["datasets"]["llama-3.3-70b__role__accountant"]
```

### 4. Quality Filtering
Select only the best examples:
```bash
# Roles: Only fully in-character (score = 3)
--min-score 3

# Traits: Only strong presence (score ≥ 70)
--min-score 70
```

## Data Provenance

- **Source**: [persona-subspace](https://github.com/lu-christina/persona-subspace) by Christina Lu
- **Models**: gemma-2-27b-it, Llama-3.3-70B-Instruct, Qwen3-32B
- **Questions**: 240 shared questions across all personas
- **Prompts**: 5 variants per persona for robustness testing
- **Quality**: GPT-4-mini scored for roleplay adherence (roles) or trait strength (traits)

## Tools

### Dataset Conversion
- Supports filtering by quality score and label (pos/neg)
- Individual dataset creation per (model, persona) pair

### Batch Processing
- Uses 16 workers for concurrent processing
- Progress tracking with tqdm

### Analysis
- `scripts/analyze_persona_token_counts.py`: Token count analysis
- Model-specific tokenizers with chat templates
- Histogram generation with matplotlib
- Score-wise token breakdowns

## Notes

- **Conversations are realistic**: Each is a genuine model response to the persona prompt
- **Scores are objective**: Rated by GPT-4-mini for consistency
- **Reproducible**: Git SHA and timestamps recorded in metadata
- **Flexible**: Filter, combine, or transform datasets as needed
- **Ready for training**: Chat format compatible with standard fine-tuning pipelines

## Citation

If you use these datasets, please cite the original persona-subspace project:

```bibtex
@misc{persona-subspace,
  author = {Christina Lu},
  title = {persona-subspace},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/lu-christina/persona-subspace}
}
```
