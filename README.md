*I'm still working on this and there is a possibility of some files not being uploaded/the readme being stale or old/irrelevant data. I'll fix them the moment I reach a certain threshhold in terms of what Phonetic can do*
# Phonetic

### A Neurosymbolic Speech Aid for People Who Stutter

[![Language](https://img.shields.io/badge/Language-Python-blue)](https://www.python.org/) [![Platform](https://img.shields.io/badge/Platform-Cross--Platform-lightgrey)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

### 🔗 Quick Links

- [💡 The Problem](#-the-problem) - [⚙️ Architecture](#️-architecture-the-neurosymbolic-pipeline) - [🚀 Getting Started](#-getting-started)
- [📐 Usage Example](#-usage-example) - [✨ Features](#-features) - [📊 Model Evaluation](#-model-evaluation) - [📂 Project Structure](#-project-structure)

---

> *"You can't fix your speech in 3 minutes, so I'll fix your speech in 3 minutes."*

---

## 💡 The Problem

People who stutter often struggle with specific phonetic onset sounds — the hard consonant clusters at the **start** of a word (e.g., `p`, `b`, `st`). These are called **phonetic blocks**. Traditional speech therapy takes months. Phonetic takes a different approach:

Instead of training the speaker, it re-writes the script.

Phonetic is a real-time, neurosymbolic pipeline that:
1. **Profiles** which onset sounds a user blocks on.
2. **Flags** every word in a target script that starts with one of those sounds.
3. **Replaces** each flagged word with a contextually natural, safe synonym — one that the user can actually say.

The output is a rewritten script the user can read fluently, right now.

([back to top](#phonetic))

---

## ⚙️ Architecture: The Neurosymbolic Pipeline

The pipeline is a two-layer system. A **symbolic layer** handles the deterministic logic (onset extraction, pattern matching, filtering), while an **ML layer** handles the ambiguous, context-sensitive decisions (which synonym actually sounds natural in a sentence).

~~~
sequenceDiagram
    participant User
    participant PangramAssessor
    participant BlockProfiler
    participant SynonymEngine
    participant ContextScorer

    User->>PangramAssessor: Reads diagnostic pangram aloud
    PangramAssessor-->>User: Extracts stutter pattern (e.g., "p-pack" → trigger: 'p')
    User->>BlockProfiler: Provides target script
    BlockProfiler-->>User: Flags words starting with trigger onsets
    BlockProfiler->>SynonymEngine: Requests safe candidates per flagged word
    SynonymEngine-->>BlockProfiler: Returns WordNet synonyms filtered by trigger list
    BlockProfiler->>ContextScorer: Scores each candidate against sentence template
    ContextScorer-->>User: Returns contextually best replacement
~~~

The `ContextScorer` is trained on 5,000 real sentences from the Brown Corpus (real English) versus 5,000 randomly shuffled word-salad sentences (unnatural English). This gives it a statistical sense of grammatical naturalness, which it uses to rank candidate synonyms.

([back to top](#phonetic))

---

## ✨ Features

### 🎙️ Dynamic Phonetic Profiling

- **Pangram Diagnostic:** The user reads a phonetically dense pangram aloud. Stutters are detected by the regex pattern `\b([a-z]{1,2})-([a-z]+)\b` — e.g., `p-pack` is parsed to extract `p` as a block trigger.
- **Onset Extraction:** The `BlockProfiler` isolates the leading consonant cluster of every word (e.g., `pl` from `planning`, `b` from `big`) and cross-references it against the user's trigger profile.
- **Interactive Mode:** The `InteractivePhonetic` class offers a full CLI-driven session — diagnostic, profiling, and live script rewriting — in a single run.

### 🔄 Safe Synonym Generation

- **WordNet Integration:** The `SynonymEngine` queries WordNet synsets for every flagged word, collecting all lemmas across all semantic senses.
- **Trigger-Aware Filtering:** Synonyms that themselves start with a user trigger are discarded before ranking. The pipeline never replaces one blocking word with another.
- **POS-Preserving Inflection:** Using `lemminflect`, replacements are inflected to match the original word's grammatical form (e.g., `present` → `exhibits`, not `exhibit`).

### 🧠 ML Context Scoring

- **Corpus-Trained Classifier:** A `CountVectorizer` (bigrams + trigrams) feeds a `LogisticRegression` model, trained to distinguish real English phrasing from unnatural word sequences.
- **Template-Based Scoring:** Each synonym candidate is inserted into the sentence template (with the original word replaced by `___`) and scored. The highest-probability natural replacement wins.
- **Brown Corpus Training:** The default `ContextScorer` trains on 5,000 sentence pairs from `nltk.corpus.brown`, the standard benchmark for English-language statistical modeling.

([back to top](#phonetic))

---

## 📊 Model Evaluation

The `ContextScorer`'s underlying classifier was evaluated on a held-out test set (20% split) of 10,000 sentence pairs.

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| **Fake / Unnatural (0)** | 0.94 | 0.93 | 0.93 |
| **Real / Natural (1)** | 0.93 | 0.94 | 0.93 |
| **Accuracy** | | | **0.93** |

The confusion matrix and per-metric plots are included as `confusion_matrix.png` and `model_metrics.png` in the repository root.

> **What this means for the pipeline:** The model correctly identifies naturally phrased English ~93% of the time. In practice, this means when it picks the "best" synonym replacement, it is picking the option that makes the sentence read most like real English — not just the closest semantic match from WordNet.

([back to top](#phonetic))

---

## 📐 Usage Example

### Full Pipeline (Scripted)

```python
from pangram_assessment import PangramAssessor
from phonetic_pipeline import PhoneticPipeline

# 1. Run a diagnostic to build the user's phonetic profile
assessor = PangramAssessor()
transcript = "p-pack my b-box with five dozen liquor j-jugs."
triggers = assessor.analyze_speech(transcript)
# => triggers: ['p', 'b', 'j']

# 2. Initialize the pipeline with the detected triggers
pipeline = PhoneticPipeline(triggers=triggers)

# 3. Rewrite a script the user needs to read
script = "I need to design a big presentation for my new job."
safe_script = pipeline.process_script(script)

# => "I need to design a large demonstration for my new occupation."
```

### Interactive CLI

```
$ python interactive_phonetic.py

============================================================
PHASE 1: PHONETIC DIAGNOSTIC
============================================================
Please read the following paragraph aloud:

'The quick brown fox jumps over the lazy dog...'

Type the words that caused a stammer or block.
Separate them with spaces (e.g., 'pack box jumps'):
> pack big

=> Profile Created! Your dynamic phonetic triggers are: ['p', 'b']

============================================================
PHASE 2: SCRIPT REWRITING
============================================================
Enter the script you want to read:
> I need to present the budget breakdown.

ORIGINAL SCRIPT: 'I need to present the budget breakdown.'
Detected Block Triggers: ['present', 'budget', 'breakdown']

Processing: 'present'
 -> Selected Replacement: 'demonstrate'

Processing: 'budget'
 -> Selected Replacement: 'financial plan'

SAFE SCRIPT: 'I need to demonstrate the financial plan breakdown.'
```

([back to top](#phonetic))

---

## 🛠️ Tech Stack Decisions

- **Why Neurosymbolic (not pure ML):** A pure ML approach would need labeled training data for every possible word substitution — impractical. A pure symbolic approach (just WordNet lookups) would produce grammatically awkward replacements with no sense of natural sentence flow. The hybrid gives the deterministic correctness of symbolic rules with the contextual judgment of a statistical model.
- **Why WordNet:** WordNet's synset graph provides semantically coherent synonyms across multiple grammatical senses. This is critical — we need replacements that preserve meaning, not just phonetic safety.
- **Why the Brown Corpus:** Brown is a balanced, general-purpose corpus covering a wide range of written English registers. Training on it gives the `ContextScorer` a broad baseline for what "natural English" looks like, rather than overfitting to a narrow domain.
- **Why Logistic Regression over a transformer:** The context-scoring task is binary (natural vs. unnatural sentence), the input is short (one sentence), and the deployment target is a local CLI with no GPU. Logistic Regression on n-gram features is interpretable, fast to train, and achieves 93% accuracy here — a transformer would add latency with no meaningful accuracy gain for this specific task.

---

## ⚠️ Known Limitations

- **Onset-Only Profiling:** The current `BlockProfiler` only flags words by their starting consonant cluster. Blocks on medial or final phonemes (e.g., `-ck` at the end of a word) are not yet handled.
- **WordNet Coverage Gaps:** Some flagged words return zero safe synonyms from WordNet. The pipeline flags these for manual review rather than silently failing.
- **No ASR Integration:** The diagnostic phase currently requires the user to manually type which words caused a block. A future version would pipe in a live ASR transcript to automate this.
- **Context Window:** The `ContextScorer` uses bigrams and trigrams, so it captures local grammatical context well but is blind to long-range sentence dependencies.

([back to top](#phonetic))

---

## 📂 Project Structure

```
/
├── interactive_phonetic.py   # 🎙️ Full CLI session (diagnostic + rewriter)
├── phonetic_pipeline.py      # 🔗 Orchestrator — wires all modules together
├── block_profiler.py         # 🚩 Onset extractor & word flagging
├── synonym_engine.py         # 📖 WordNet synonym lookup + trigger filtering
├── context_scorer.py         # 🧠 Brown Corpus-trained naturalness classifier
├── pangram_assessment.py     # 📋 Stutter pattern detection from transcript
├── ml_engine.py              # 🤖 Intent classifier for fragmented AAC-style input
├── data_prep.py              # 🗂️ Synthetic training data + text normalizer
├── evaluate_model.py         # 📊 Holdout evaluation + confusion matrix export
├── confusion_matrix.png      # 📉 Model evaluation output
└── model_metrics.png         # 📈 Precision / Recall / F1 visualization
```

([back to top](#phonetic))

---

## 🚀 Getting Started

**Prerequisites**

- Python 3.10+
- pip

**Install dependencies**

```bash
pip install nltk scikit-learn pandas matplotlib seaborn lemminflect
```

**Run the interactive session**

```bash
python interactive_phonetic.py
```

**Run the full pipeline (scripted)**

```bash
python phonetic_pipeline.py
```

**Evaluate the context scorer**

```bash
python evaluate_model.py
```

NLTK corpora (`brown`, `wordnet`, `webtext`, `punkt`, `averaged_perceptron_tagger`) are downloaded automatically on first run if not already present.

---

## 👨‍💻 Author

Built by **Ameetesh**  
B.Tech Undergraduate (South Asian University)  
Focused on NLP, Assistive Technology, and Neurosymbolic AI.

---

## License

MIT.

([back to top](#phonetic))
