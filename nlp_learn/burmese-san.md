# BURMESE-SAN: Burmese NLP Benchmark for Evaluating Large Language Models
### Thura Aung · Jann Railey Montalan · Jian Gang Ngui · Peerat Limkonchotiwat
#### LREC 2026 | AI Singapore · NUS · KMITL
> 🌐 Leaderboard: https://leaderboard.sea-lion.ai/detailed/MY

---

## Slide 1: Motivation — Why Burmese?

- **43–45 million** speakers, official language of Myanmar *(Ethnologue, 2019)*
- Still **under-resourced** for NLP — scarce data, tools, and benchmarks *(Dou et al., 2025)*
- Existing multilingual benchmarks (**HELM, SEA-HELM**) **exclude Burmese** due to lack of available datasets

### What makes Burmese linguistically challenging?
| Feature | Description |
|---|---|
| Word order | Subject-Object-Verb (SOV) |
| Morphology | Rich, agglutinative morphology |
| Tones | 4 distinct tones — phonemically distinctive |
| Diglossia | Formal vs. informal registers differ significantly |
| Script | Unique Myanmar script — distinct from Latin/CJK |

> *"Burmese features an extensive case-marking system and a complex honorifics structure reflecting social hierarchy."* *(Jenny, 2021)*

---

## Slide 2: Problem Statement & Research Gap

### Existing benchmarks largely exclude Burmese

| Benchmark | Languages Covered | Burmese? |
|---|---|---|
| HELM *(Liang et al., 2023)* | English | ✗ |
| SEA-HELM *(Susanto et al., 2025)* | Multiple SEA languages | ✗ |
| SeaExam / SeaBench *(Liu et al., 2025)* | Vietnamese, Thai, Indonesian | ✗ |
| BHASA *(Leong et al., 2023)* | Indonesian, Tamil, Thai, Vietnamese | ✗ |
| SeaEval *(Wang et al., 2024)* | 29 datasets (multilingual) | ✗ |

### Key Problems
1. No unified benchmark exists for evaluating LLMs on Burmese
2. Prior Burmese data relies on **machine translation** — unnatural, disfluent
3. No coverage of **generation** or **reasoning** capabilities

---

## Slide 3: Introducing BURMESE-SAN

> ***"San"*** means ***"Standard"*** in Burmese.

**BURMESE-SAN** = First **holistic** Burmese benchmark for evaluating LLMs

### Three Core NLP Competencies

```
┌──────────────────────────────────────────────────────┐
│  NLU (Understanding)  │  NLR (Reasoning)  │  NLG (Generation)  │
│  QA · SA · TD         │  CR · NLI         │  AS · MT           │
└──────────────────────────────────────────────────────┘
```

### Key Properties
- **7 tasks**, **3,920 samples**, balanced class distributions
- **Human-centered pipeline** — native speakers involved at every stage
- **Culturally authentic** — reflects real-world Burmese language use
- **Public leaderboard** for ongoing evaluation

---

## Slide 4: Benchmark Tasks at a Glance

| Competency | Task | Dataset | Source | # Samples |
|---|---|---|---|---|
| **NLU** | Question Answering (QA) | Belebele *(Bandarkar et al., 2024)* | English-adapted → Native re-translation | 120 |
| **NLU** | Sentiment Analysis (SA) | GKLMIP-mya *(Koto et al.)* | Natively-sourced | 600 (3 classes) |
| **NLU** | Toxicity Detection (TD) | myHateSpeech *(Ye et al.)* | Natively-sourced | 400 (2 classes) |
| **NLR** | Causal Reasoning (CR) | Balanced COPA *(Kavumba et al., 2019)* | English-sourced → Native translation | 400 (2 classes) |
| **NLR** | Natural Language Inference (NLI) | myXNLI *(Htet et al., 2025)* | Natively-translated | 600 (3 classes) |
| **NLG** | Abstractive Summarization (AS) | XL-Sum *(Hasan et al., 2021)* | English-adapted → Native re-translation | 100 |
| **NLG** | Machine Translation (MT) | FLORES+ *(NLLB, 2024)* | English-adapted → Native re-translation | 1,700 (bi-directional) |

---

## Slide 5: Dataset Curation Pipeline

### Four-Step Process

```
Step 1          Step 2              Step 3                  Step 4
Random      ──► Translation    ──► Normalization &     ──► Native
Sampling &      + Binary            Label Verification      Revision
Filtering       Criteria Rating
```

### Annotator Profile
- Native Burmese speakers, ages **18–25**
- University students enrolled in **international programs**
- Involved in: task selection → prompt translation → annotation → QA

### Quality Criteria Used
| Criterion | Applied to | Definition |
|---|---|---|
| Completeness | All tasks | All source info retained in translation |
| Fluency | All tasks | Grammatically correct and natural Burmese |
| Sensibility | All tasks | Logically appropriate given context |
| Faithfulness | MT | Meaning preserved from original |
| Relevance | AS | Summary contains most important content |
| Coherence | AS | Summary is logically structured |

**Initial overall joint agreement: 99.11%**

---

## Slide 6: Dataset Quality — Inter-Annotator Agreement

### Label Verification Results (SA and TD)
| Task | Cohen's Kappa | Krippendorff's Alpha | Total Agreement |
|---|---|---|---|
| Sentiment Analysis (SA) | 0.948 | 0.948 | 96.5% |
| Toxicity Detection (TD) | 0.985 | 0.985 | 99.25% |

### Translation Quality (Before Native Revision)
| Task | Criterion | Joint Agreement (%) |
|---|---|---|
| QA | Completeness | 100.00 |
| QA | Fluency | 97.50 |
| MT | Faithfulness | 99.88 |
| AS | All criteria | 100.00 |
| CR | Fluency | 97.00 |

> Samples without **full consensus** were manually revised by native speakers before inclusion.

---

## Slide 7: Challenges in Building a Burmese Benchmark

### Challenge 1: Inconsistency in Technical Terms
- No widely adopted scientific vocabulary standard *(Myanmar Sci. Dict., 2019)*
- Different translators use different transliterations (e.g., "Beijing", "Theoretical")

### Challenge 2: Syntax Variation
- Burmese is flexible — word order can vary while preserving meaning
- Multiple particles with overlapping functions → stylistic divergence

### Challenge 3: Machine Translation Failures
- Google Translate output is **grammatically passable but unnatural**
- Literal translation of idioms → reversed/lost meanings
- Issues found in: FLORES+, Belebele, XL-Sum original translations

### Our Solution
- **Cross-revision** by native speakers: each annotator reviewed others' work
- Applied **retranslation** rather than post-editing for most tasks
- Preserved code-switching (English-Burmese) in SA and TD — natural in Myanmar social media

---

## Slide 8: Experimental Setup

### Evaluation Metrics
| Competency | Tasks | Metric |
|---|---|---|
| NLU / NLR | QA, SA, TD, CR, NLI | Accuracy |
| NLG | MT | MetricX-24 *(Juraska et al., 2024)* |
| NLG | AS | ROUGE-L F1 (multilingual, from XL-Sum) |

All scores **normalized to [0, 100]** following SEA-HELM normalization.

### Prompting Strategy
- **Zero-shot** prompting (no in-context examples)
- All prompt templates written entirely in **formal Burmese**
- Templates adapted from SEA-HELM *(Susanto et al., 2025)* — clear separation of instruction and content

### Evaluation Details
- **8 independent runs** per model (non-greedy, temperature > 0)
- Final score = **mean** across runs for stability

### Models Evaluated
- **Instruction-tuned** models (Qwen 3, Llama 3/4, Gemma 2/3, SEA-LION, DeepSeek, etc.)
- **Reasoning** models (Qwen 3 Thinking, DeepSeek R1, GPT-OSS, SEA-LION R, etc.)
- **Commercial** models (Gemini 2/2.5, GPT-4o/4.1/5, Claude Sonnet 4, Opus 4.1)

---

## Slide 9: Results — Instruction-Tuned Models (Top 3 per Size)

> Score = normalized MY (overall Burmese score). Best in **bold**, 2nd **underlined**.

### Small Models (< 14B)
| Rank | Model | MY | CR | NLI | QA | SA | TD | AS | MT |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **Gemma 3 VL (12B)** | **42.46** | **44.93** | 26.22 | **67.11** | **48.96** | **40.97** | 18.77 | **70.96** |
| 🥈 | SEA-LION v4-Qwen-3-VL (8B) | 30.67 | **38.77** | 23.28 | **58.22** | 36.21 | 31.76 | 15.49 | 51.70 |
| 🥉 | Qwen 3 VL (8B) | 30.25 | 36.85 | **32.05** | 53.59 | 36.41 | **32.60** | 16.51 | 49.72 |

### Medium Models (14B–32B)
| Rank | Model | MY | CR | NLI | QA | SA | TD | AS | MT |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **SEA-LION v4-Qwen-3-VL (32B)** | **49.56** | **66.38** | **57.13** | **81.52** | 40.49 | **51.70** | 23.56 | 64.00 |
| 🥈 | Gemma 3 VL (27B) | 48.14 | 57.08 | **53.82** | 69.85 | 47.13 | **49.96** | 13.16 | **79.44** |
| 🥉 | SEA-LION v4-Gemma-3-VL (27B) | 47.18 | 57.15 | 52.92 | 69.11 | **47.56** | 48.95 | 11.43 | **77.83** |

### Large Models (> 32B)
| Rank | Model | MY | CR | NLI | QA | SA | TD | AS | MT |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **ERNIE 4.5 (300B MoE)** | **54.68** | 68.82 | 40.39 | **78.52** | **49.81** | **54.27** | 20.84 | **86.25** |
| 🥈 | Qwen 3 A22B (235B MoE) | **54.29** | **71.32** | **57.42** | 76.63 | **50.32** | **56.51** | 23.36 | 78.40 |
| 🥉 | Llama 4 Maverick (400B MoE) | 51.49 | 67.87 | 31.87 | **80.00** | 44.92 | 52.54 | **30.66** | 81.86 |

---

## Slide 10: Results — Reasoning Models (Top 3 per Size)

### Small Reasoning Models (< 14B)
| Rank | Model | MY | CR | NLI | QA | SA | TD | AS | MT |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **Qwen 3 Thinking (8B)** | **37.12** | **55.80** | **33.45** | **65.19** | **40.18** | **40.22** | **18.28** | **59.53** |
| 🥈 | Qwen 3 Thinking (4B) | 34.78 | 40.78 | 31.98 | 60.37 | 31.82 | 35.77 | 16.29 | 56.00 |
| 🥉 | SEA-LION v3.5 R Llama (8B) | 22.08 | 0.00 | 30.19 | 54.00 | 0.00 | 22.28 | 11.02 | 48.28 |

### Medium Reasoning Models (14B–32B)
| Rank | Model | MY | CR | NLI | QA | SA | TD | AS | MT |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **Qwen 3 Thinking (32B)** | **52.35** | **71.32** | **48.61** | **77.33** | **42.32** | **48.49** | **22.90** | 43.43 |
| 🥈 | Qwen 3 Thinking (14B) | 43.81 | 61.45 | 35.47 | 73.70 | 38.71 | 45.07 | 20.86 | **65.04** |
| 🥉 | QwQ (32B) | 24.66 | 0.00 | 19.04 | 55.96 | 30.38 | 18.58 | 4.17 | 38.97 |

### Large Reasoning Models (> 32B)
| Rank | Model | MY | CR | NLI | QA | SA | TD | AS | MT |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **Qwen 3 Thinking (235B MoE)** | **66.39** | **77.97** | **51.53** | **82.44** | **44.68** | **61.37** | 20.48 | 85.48 |
| 🥈 | DeepSeek V3.1 Thinking (671B MoE) | 59.46 | 71.65 | 48.27 | 78.26 | 40.91 | **57.75** | 21.21 | **86.46** |
| 🥉 | Qwen 3 Next Thinking (80B MoE) | 57.97 | 72.50 | 49.68 | 80.04 | 40.75 | 56.35 | 20.08 | 79.88 |

---

## Slide 11: Results — Commercial Models

> No size sub-groups; all proprietary APIs.

| Rank | Model | MY | CR | NLI | QA | SA | TD | AS | MT |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **Gemini 2.5 Pro** | **72.35** | **85.75** | 67.55 | **85.48** | 45.56 | **68.74** | 24.21 | **90.22** |
| 🥈 | Gemini 2.5 Flash | 69.34 | **83.58** | 61.07 | **84.15** | 39.60 | **66.16** | 24.11 | **89.49** |
| 🥉 | GPT-5 (2025-08-07) | 66.46 | 79.87 | 43.63 | 81.37 | 45.70 | 60.16 | 17.09 | 87.46 |
| 4th | Gemini 2 Flash | 59.23 | 76.58 | 41.30 | 79.81 | 41.48 | 57.78 | 22.09 | 88.23 |
| 5th | GPT-4.1 | 55.80 | 68.80 | 42.80 | 77.15 | **50.24** | 54.05 | 21.37 | 79.73 |
| 6th | GPT-4o | 51.61 | 70.77 | 39.48 | 75.67 | **50.94** | 52.34 | 21.97 | 78.58 |
| 7th | Opus 4.1 | 45.10 | 71.57 | 0.00 | 0.00 | 48.35 | 45.66 | 24.69 | 87.52 |
| 8th | Sonnet 4 | 40.45 | 24.33 | **86.01** | 0.00 | 47.82 | 43.22 | **68.10** | 0.00 |

> ⚠️ Sonnet 4 achieves the **highest NLI (86.01%)** and **AS (68.10%)** but fails on QA and MT — format compliance issue.

---

## Slide 12: Key Findings

### Finding #1 — Commercial > Open-weight (RQ1)
- **Gemini 2.5 Pro (72.35%)** leads overall, followed by Gemini 2.5 Flash and GPT-5
- Gap of ~**17.67%** between top commercial and best open-weight (ERNIE 4.5, 54.68%)
- Gap is widest in culturally-specific reasoning tasks

### Finding #2 — Scale Helps, but Is Not Sufficient (RQ2)
- Larger models generally outperform smaller ones **within the same family**
- Notable exceptions: Kimi K2 (1040B MoE, 43.94%) < Qwen 3 32B (52.35%)
- **Architecture, training data, and tuning matter more than parameter count alone**

### Finding #3 — SEA Fine-tuning Is Architecture-Dependent (RQ3)
- **Llama-based SEA-LION** gains most from SEA fine-tuning (especially at 70B)
- Qwen-based: moderate +4.96%; Gemma-based: slight degradation −0.96%
- Largest task-level gains: **MT (+19.29%)**, **QA (+7.04%)**

### Finding #4 — Quantization Is Generally Safe (RQ4)
- **DynFP8** quantization largely preserves performance across model families
- **NVFP4** (4-bit) causes notable degradation, especially for reasoning models
- Effect varies by task: reasoning-heavy tasks degrade more under aggressive quantization

### Finding #5 — Rapid Generational Improvement (RQ5)
- Llama: 3.3 70B (23.07%) → Llama 4 Maverick (51.49%) → **+28.42pp**
- GPT: 4o (51.61%) → GPT-5 (66.46%) → **+14.85pp**
- Gemini: 2 Flash (59.23%) → 2.5 Pro (72.35%) → **+13.12pp**
- SEA-LION: v3 Gemma-2 9B (15.40%) → v4 Gemma-3 4B (26.24%)

---

## Slide 13: Overall Performance Summary

```
Overall Burmese Score (MY) — Best per Category

Instruction-tuned (Small)  : Gemma 3 VL 12B          → 42.46
Instruction-tuned (Medium) : SEA-LION v4-Qwen-3 32B  → 49.56
Instruction-tuned (Large)  : ERNIE 4.5 300B MoE       → 54.68

Reasoning (Small)          : Qwen 3 Thinking 8B       → 37.12
Reasoning (Medium)         : Qwen 3 Thinking 32B      → 52.35
Reasoning (Large)          : Qwen 3 Thinking 235B MoE → 66.39

Commercial                 : Gemini 2.5 Pro            → 72.35
```

> **Performance bottleneck**: Abstractive Summarization (AS) remains the hardest task across all model types and sizes — generative fluency in Burmese lags behind.

---

## Slide 14: Conclusion

### What We Proposed
- **BURMESE-SAN** — first holistic LLM benchmark for Burmese
- 7 tasks across NLU, NLR, NLG → **3,920 high-quality samples**
- Fully human-verified, native-speaker-driven curation pipeline

### What We Found
- Commercial models lead, but open-weight models are catching up
- **Architecture + tuning > scale** for Burmese performance
- SEA-specific fine-tuning especially benefits **Llama-based** architectures
- Rapid acceleration observed across model generations

### BURMESE-SAN as a Foundation
- Provides a **standardized, reproducible** platform for Burmese NLP evaluation
- Supports **automated tracking** via public leaderboard
- Enables principled comparison of future Burmese-focused models

---

## Slide 15: Limitations & Future Work

### Current Limitations
- Prompts use **formal written Burmese only** — colloquial register not evaluated
- Covers **Standard Burmese** (central Myanmar) — excludes dialects:
  - Arakanese (Rakhine), Tavoyan, Intha, Yaw, Merguese, Palaw
- No multimodal or speech evaluation

### Future Directions
1. Add **colloquial and spoken-style** prompt templates
2. Extend coverage to **Burmese dialects** (community-driven data collection)
3. Evaluate **cross-lingual** transfer and few-shot prompting settings
4. Explore **Burmese-specific pretraining** data augmentation strategies

---

## Slide 16: References (Selected)

| Citation | Work |
|---|---|
| Liang et al., 2023 | HELM: Holistic Evaluation of Language Models |
| Susanto et al., 2025 | SEA-HELM: Southeast Asian Holistic LLM Benchmark |
| Lovenia et al., 2024 | SEACrowd: Southeast Asian NLP Community |
| Bandarkar et al., 2024 | Belebele: Multilingual QA Dataset |
| Htan et al., 2025 | myXNLI: Burmese NLI Dataset |
| Kavumba et al., 2019 | Balanced COPA: Causal Reasoning |
| Hasan et al., 2021 | XL-Sum: Multilingual Summarization |
| NLLB, 2024 | FLORES+: Multilingual Translation Benchmark |
| Dou et al., 2025 | Sailor2: Sailing Southeast Asia LLM |
| Ng et al., 2025 | SEA-LION: Southeast Asian LLM |
| Yang et al., 2025 | Qwen3 Technical Report |
| DeepSeek AI, 2025 | DeepSeek-V3 / R1 Technical Reports |
| Grattafiori et al., 2024 | Llama 3 Herd of Models |
| Gemini Team, 2025 | Gemini Family Report |

---

> **Dataset & Leaderboard**: https://leaderboard.sea-lion.ai/detailed/MY
> **License**: CC BY-SA 4.0
