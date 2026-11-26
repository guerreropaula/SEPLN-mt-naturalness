# Naturalness in LLM Translation for Low-Resource Languages


This repository contains the supplementary material for the paper **“Naturalness in LLM Translation for Low-Resource Languages”**.

The study investigates translation naturalness in Large Language Models (LLMs) for **English-to-Basque (EN→EU)** and **English-to-Catalan (EN→CA)**. It compares interventions of increasing complexity, including **prompting**, **supervised fine-tuning (SFT)**, and **reinforcement learning with Group Relative Policy Optimization (GRPO)**, and evaluates their effect on both translation quality and different dimensions of naturalness.

The repository contains the code and detailed results that could not be included in the paper due to space limitations.


## CONTENTS

- **`configs/`**: experiment, model, SFT, GRPO, and evaluation configurations.
- **`results/`**: complete results for the in-domain and out-of-domain evaluations, including FLORES+ and the literary test set.
- **`src/data/`**: corpus preprocessing and dataset construction.
- **`src/prompting/`**: implementation of the P0–P3 prompting experiments.
- **`src/sft/`**: supervised fine-tuning (P4).
- **`src/grpo/`**: GRPO training, reward functions, ablations, and evaluation (P5).
- **`src/classifiers/`**: discriminative naturalness estimators.
- **`src/contrastive_lm/`**: contrastive language-model naturalness estimators.
- **`src/evaluation/`**: translation-quality and naturalness evaluation, comparison scripts, and statistical analysis.


The main datasets and external resources used are:

- **English→Basque:** [EHU-HAC (Hizkuntzen Arteko Corpusa)](https://www.ehu.eus/ehg/hac/).
- **English→Catalan:** [AINA CA-EN Parallel Corpus](https://huggingface.co/datasets/projecte-aina/CA-EN_Parallel_Corpus), restricted to the `HRM` and `CUL` domains.
- **General-domain external evaluation:** [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus).
- **Literary evaluation:** Virginia Woolf's *To the Lighthouse* and its published Basque and Catalan translations:
  - Basque: *Farorantz*, translated by Anton Garikano, available through [Armiarma's liburu-e collection](https://armiarma.eus/liburu-e/).
  - Catalan: *Cap al far*, translated by Xavier Pàmies; the digital edition used in the study was obtained from [PocketBook](https://pocketbook.es/cap-al-far-9788498594799).
- **Synonym-frequency analysis:** [Apertium English–Catalan](https://github.com/apertium/apertium-eng-cat) and [Apertium Basque–English](https://github.com/apertium/apertium-eu-en) bilingual dictionaries.



