# PII Detection and Text Anonymization in Legal Documents

**Detecting and masking personal information in court decisions without making them useless**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![BERT](https://img.shields.io/badge/Model-bert--base--cased-2a78d6)
![spaCy](https://img.shields.io/badge/spaCy-NER-09A3D5)
![GPU](https://img.shields.io/badge/GPU-T4%20(Colab)-76B900?logo=nvidia&logoColor=white)

| Best F1 | Identifiers masked | Over-masking | Dataset |
|:---:|:---:|:---:|:---:|
| **0.81** (BERT) vs 0.44 (spaCy) | **85%** | **19%** vs 85% (spaCy) | 1,268 ECHR court judgments |

---

## Problem

Court decisions are published for transparency, but they contain personal data that must be anonymized under privacy laws such as the GDPR. Manual anonymization is slow and cannot keep up with the volume of documents.

The real difficulty is not detecting names: it is deciding **which entities must be hidden**. A judge's name or a government body can stay visible, while an applicant, a witness or a precise date that could re-identify someone must be masked. Standard NER tools detect entity *types* but cannot make that distinction.

> **Research question:** can a model that understands context decide which entities to mask, protecting privacy while keeping the document useful?

## State of the art

| Approach | Example | Limitation |
|---|---|---|
| Rules + NER toolkits | Microsoft Presidio | Poor generalization to legal text |
| Specialized PII models | GLiNER-based PII detectors | Detect the entity type, not whether it should be hidden |
| Legal fine-tuned models | Models trained on court cases (TAB) | People can remain re-identifiable through indirect clues |
| LLM-based anonymizers | Staab et al. (ICLR 2025) | Costly, and sensitive data must be processed by large models |

## Dataset

**[Text Anonymization Benchmark (TAB)](https://github.com/NorskRegnesentral/text-anonymization-benchmark)**: English judgments from the European Court of Human Rights, annotated by law students specifically for anonymization.

| | |
|---|---|
| Documents | 1,268 court judgments |
| Entity mentions | 155,006 |
| Entity types | PERSON, ORG, LOC, DATETIME, CODE, DEM, QUANTITY, MISC |
| Masking labels | `DIRECT` identifier, `QUASI` identifier, `NO_MASK` |
| Split | 1,014 train / 127 validation / 127 test documents |

## Data preparation

Anonymization requires the **exact original text** (annotations are character positions, and capitalization is a key clue), so no lowercasing, stopword removal or lemmatization is applied.

1. **Target definition**: entities that must be masked (`DIRECT` + `QUASI`), one annotator per document.
2. **BIO labeling**: spans converted into token labels `B-MASK`, `I-MASK`, `O`.
3. **Tokenization and alignment** with a **cased** BERT tokenizer.
4. **Chunking**: 512-token windows with a 128-token overlap so that no entity is lost (4,771 / 395 / 378 windows).

> Example: "Mr John Smith lives in Rome" → `O B-MASK I-MASK O O B-MASK`

## Methodology

Anonymization is framed as **token classification**. Three models of increasing complexity are compared on the same test set with the same metrics:

| Model | What it tests |
|---|---|
| **Baseline 1: spaCy NER**, masks every detected entity | Is detecting entities enough? (no context, no training) |
| **Baseline 2: hand-crafted features + Logistic Regression** | Can a classic ML model learn *which* entities to mask? |
| **BERT (`bert-base-cased`) fine-tuned on TAB** | Does contextual understanding improve the decision? |

**Why `bert-base-cased`:** it keeps capitalization (a key clue for names), unlike uncased models and LEGAL-BERT, and it trains in about 30 minutes on a free T4 GPU, unlike Longformer.

**Fine-tuning:** learning rate 3e-5, effective batch size 16 (8 × 2 gradient accumulation), 3 epochs, 10% warmup, weight decay 0.01, mixed precision, best epoch selected on validation F1. On full documents, predictions from overlapping windows are averaged.

**Metrics:** entity-level precision, recall and F1, plus two task-specific measures:
- **Privacy**: share of identifiers fully masked (higher is better).
- **Utility**: share of entities that should stay visible but were masked (lower is better).

## Results

| Model | Precision | Recall | F1 | Identifiers masked ↑ | Kept entities wrongly masked ↓ |
|---|:---:|:---:|:---:|:---:|:---:|
| spaCy NER (mask all) | 0.33 | 0.67 | 0.44 | 76% | 85% |
| Features + Logistic Regression | 0.70 | 0.73 | 0.71 | 75% | 23% |
| **BERT fine-tuned** | **0.80** | **0.82** | **0.81** | **85%** | **19%** |

- **Accuracy is misleading here**: 89% of words do not need masking, so a model that masks nothing already scores high.
- **spaCy** masks 85% of the entities that should stay visible: detecting entities is not enough.
- **Logistic Regression** learns from TAB which entities to keep and cuts over-masking from 85% to 23%.
- **BERT** is the best on every metric: it protects **more** identifiers (85%) **and** masks **fewer** entities by mistake (19%). Context is the key.

### Error analysis

- BERT rarely misses clear identifiers (persons, dates, codes: about 5% missed), but often misses vague ones such as miscellaneous (90%) and demographic (58%) entities.
- It learned to keep organizations and demographic attributes visible, but still masks **93% of person names that should be kept**: telling a public person (judge, government agent) from a private one remains the hardest part.

## Limitations

- **One annotator per document**: other annotators may have made different, valid choices, and the same person can be "keep" in one document and "mask" in another.
- **Limited context**: BERT reads 512 tokens at a time and can miss a person's role explained earlier in the document.
- **Mentions handled separately**: missing a single mention of a name can be enough to re-identify someone.
- **No re-identification test**: real privacy after anonymization was not measured.

**Future work:** long-document models (Longformer), consistent masking of all mentions of the same person, and re-identification attacks to measure actual privacy.

## Structure

```
legal-pii-anonymization-bert/
├── NLP_Anonymization_Project.ipynb   ← full notebook (data, baselines, BERT, results, error analysis)
└── README.md
```

## Reproduce

1. Open the notebook in Google Colab with a **T4 GPU**.
2. Run all cells: the TAB dataset is downloaded automatically, and BERT fine-tuning takes about 30 minutes.

## References

1. I. Pilán et al., "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization," *Computational Linguistics*, 48(4), 2022.
2. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *NAACL*, 2019.
3. R. Staab, M. Vero, M. Balunović, M. Vechev, "Large Language Models are Advanced Anonymizers," *ICLR*, 2025.
4. Microsoft Presidio, data protection and de-identification SDK.

---

**Author:** Eya Bouhmida · [GitHub](https://github.com/eya-bouhmida)
