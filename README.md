Perfect — thanks for clarifying. Here’s your updated and **final clean README** version reflecting your current project state:
(no `models/` folder, only `models_scratch/` exists).

---

# LM-Corpus-Effects

Exploring the Effect of Training Data on BERT's Generalization

This project investigates how different training corpora affect a language model’s ability to generalize and predict missing or next words.
It trains and evaluates BERT-style models (from scratch) using:

* A full corpus (corpus_original.txt) containing a few hundred short documents across diverse topics.
* A filtered corpus (corpus_filtered.txt) which is the same dataset but with specific keywords and topics removed, such as geography, sports, and technology.

The experiment is based on the idea of Filtered Corpus Training (FiCT), which studies how removing linguistic evidence from the training data changes model performance on unseen prompts.

---

Project Goals

1. Build two corpora (original vs. filtered) using Hugging Face datasets.
2. Train small BERT models from scratch on both corpora.
3. Evaluate their performance on masked word and next-word prediction tasks.
4. Analyze how filtering affects generalization, accuracy, and prediction diversity.

---

Repository Structure

lm-corpus-effects/
├── data/
│   ├── corpus_original.txt
│   ├── corpus_filtered.txt
│   ├── eval.txt
│   ├── nextword_eval.txt
│   ├── nextword_answers.tsv
│   └── README.md
│
├── ds/
│   ├── corpus_original/          # Tokenized dataset caches
│   ├── corpus_filtered/          # Filtered dataset splits
│   └── eval/                     # Evaluation datasets
│
├── models_scratch/
│   ├── corpus_original_tiny/
│   └── corpus_filtered_tiny/
│
├── src/
│   ├── data/
│   │   ├── make_corpora.py
│   │   ├── prepare_dataset.py
│   │   ├── train_tokenizer.py
│   │
│   ├── train/
│   │   └── train_bert_scratch.py
│   │
│   └── eval/
│       ├── eval_bert.py
│       ├── make_nextword_eval.py
│       └── score_nextword_bert.py
│
├── tokenizer/
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── requirements.txt
├── tiny_bert_config.json
└── README.md

---

Folder Overview

data/
Contains all corpus and evaluation text files used for training and testing.

ds/
Holds preprocessed Hugging Face datasets (tokenized and split versions of the corpora). Useful for quick reloading instead of re-tokenizing from scratch.

models_scratch/
Contains the trained BERT models built from random initialization.
Each subfolder represents a different training corpus (original vs. filtered).

tokenizer/
Stores the tokenizer vocabulary and configuration used for all scratch-trained models.

---

Data Overview

corpus_original.txt - Full training corpus (~600 short documents).
corpus_filtered.txt - Filtered version (~530 documents) with targeted keywords removed.
eval.txt - Handwritten fill-in-the-blank prompts for manual evaluation.
nextword_eval.txt - Automatically generated next-word prediction prompts.
nextword_answers.tsv - Gold-standard answers for next-word evaluation.

---

Code Components

src/data/

* make_corpora.py: Builds the original and filtered corpora from public datasets, cleans text, and removes unwanted keywords.
* prepare_dataset.py(Optional/not used): Converts text data into tokenized Hugging Face Dataset objects for efficient training. // not used anymore
* train_tokenizer.py(Optional/not used): Trains or loads a BertTokenizerFast vocabulary shared across all experiments. // not used anymore
// train_bert_scratch does everything inline it uses bert-base-uncased tokenizer, tokenizes the text dynamically via build datasets

src/train/

* train_bert_scratch.py: Builds and trains a tiny, randomly initialized BERT model entirely from scratch on the given corpus. This demonstrates how limited data and topic filtering affect a model’s ability to learn structure and meaning.

src/eval/

* eval_bert.py: Evaluates a trained model on fill-in-the-blank (masked token) prompts and prints the top-5 predictions for each.
* make_nextword_eval.py: Generates next-word prediction prompts and corresponding gold answers from the training corpus.
* score_nextword_bert.py: Scores models based on top-1 and top-5 accuracy, and average probability assigned to the correct word.

---

Metrics Explained

acc@1 - The percentage of prompts where the model’s top prediction matches the correct next word.
acc@5 - The percentage where the correct word appears in the model’s top five predictions.
avg_gold_prob - The average probability the model assigns to the correct next word across all prompts.

---

Evaluation Workflow

1. Build corpora:
   python src/data/make_corpora.py

2. Train models from scratch:
   python src/train/train_bert_scratch.py data/corpus_original.txt models_scratch/corpus_original_tiny
   python src/train/train_bert_scratch.py data/corpus_filtered.txt models_scratch/corpus_filtered_tiny

3. Evaluate fill-in-the-blank predictions:
   python src/eval/eval_bert.py models_scratch/corpus_original_tiny
   python src/eval/eval_bert.py models_scratch/corpus_filtered_tiny

** Ask Professor Lin 

4. Evaluate next-word prediction:
   python src/eval/make_nextword_eval.py data/corpus_original.txt --out_prompts data/nextword_eval.txt --out_answers data/nextword_answers.tsv --max_items 500
   python src/eval/score_nextword_bert.py models_scratch/corpus_original_tiny data/nextword_eval.txt data/nextword_answers.tsv
   python src/eval/score_nextword_bert.py models_scratch/corpus_filtered_tiny data/nextword_eval.txt data/nextword_answers.tsv

---

Interpreting the Results

These models are trained on very small datasets and are not expected to perform well in absolute accuracy.
The goal is not to achieve state-of-the-art performance, but to compare how the filtered corpus affects model behavior and generalization.

A well-trained model on corpus_original will typically show slightly higher next-word and masked-word accuracy than the filtered version, especially on prompts related to the removed topics.

This demonstrates the concept that linguistic coverage and context in training data directly influence model generalization ability.

---

Next Steps

1. Expand or rebalance the corpus to test larger or more selective filters.
2. Visualize vocabulary overlap and token distributions between corpora.
3. Add quantitative plots comparing acc@1 and acc@5 between models.
4. Experiment with training durations, batch sizes, or masking probabilities.
5. Document qualitative examples showing how predictions differ by corpus.

---

Maintainer: Erick Mainoo , Alex Geer, Mollie Hamman
Course: CS 5/7330 - Natural Language Processing, Fall 2025
Institution: Southern Methodist University (SMU), Dallas TX
Last Updated: October 2025

---

**short 150-word project summary paragraph** 
