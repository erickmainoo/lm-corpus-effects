
# LM-Corpus-Effects

Exploring the Effect of Training Data on BERT's Generalization

This project investigates how different training corpora affect a language model’s ability to generalize and predict missing or next words.
It trains and evaluates BERT-style models (from scratch) using:

* A full corpus (`corpus_original.txt`) containing a few hundred short documents across diverse topics.
* A filtered corpus (`corpus_filtered.txt`) which is the same dataset but with specific keywords and topics removed, such as geography, sports, and technology.

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
│   
│
├── ds/
│   ├── corpus_original/
│   ├── corpus_filtered/
│   └── eval/
│
├── models_scratch/
│   ├── corpus_original_tiny/
│   └── corpus_filtered_tiny/
│
├── src/
│   ├── data/
│   │   ├── make_corpora.py
│   │   ├── prepare_dataset.py
│   │   └── train_tokenizer.py
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

corpus_original.txt – Full training corpus (~600 short documents).
corpus_filtered.txt – Filtered version (~530 documents) with targeted keywords removed.
eval.txt – Handwritten fill-in-the-blank prompts for manual evaluation.
nextword_eval.txt – Automatically generated next-word prediction prompts.
nextword_answers.tsv – Gold-standard answers for next-word evaluation.

---

Code Components

src/data/

* make_corpora.py: Builds and cleans the original and filtered corpora, removing selected keywords and saving the final text datasets.
* prepare_dataset.py: (Optional / not currently used) Converts text files into pre-tokenized Hugging Face Dataset objects. Kept for possible future use.
* train_tokenizer.py: (Optional / not currently used) Trains a new tokenizer vocabulary from the small corpus; currently the pretrained “bert-base-uncased” tokenizer is reused instead.

src/train/

* train_bert_scratch.py: Builds and trains a tiny, randomly initialized BERT model entirely from scratch on the given corpus. This demonstrates how limited data and topic filtering affect a model’s ability to learn structure and meaning.

src/eval/

* eval_bert.py: Evaluates a trained model on fill-in-the-blank (masked token) prompts and prints the top-5 predictions for each.
* make_nextword_eval.py: Generates next-word prediction prompts and corresponding gold answers from the training corpus.
* score_nextword_bert.py: Scores models based on top-1 and top-5 accuracy, and average probability assigned to the correct word.

---

How to Run Each File

1. Build the corpora

   ```
   python src/data/make_corpora.py
   ```

   Creates two small text datasets:

   * data/corpus_original.txt (full)
   * data/corpus_filtered.txt (filtered)
     You can optionally specify output paths with --out_original and --out_filtered.

2. Train a model from scratch

   ```
   python src/train/train_bert_scratch.py data/corpus_original.txt models_scratch/corpus_original_tiny --epochs 8 --batch_size 16 --max_length 128
   ```

   Trains a small BERT model from random initialization on your corpus and saves it under models_scratch/corpus_original_tiny.

   Key parts:

   * data/corpus_original.txt – input corpus
   * models_scratch/corpus_original_tiny – output model directory
   * --epochs – number of training passes
   * --batch_size – examples per training step
   * --max_length – maximum token sequence length

   To train on the filtered version:

   ```
   python src/train/train_bert_scratch.py data/corpus_filtered.txt models_scratch/corpus_filtered_tiny
   ```

3. Evaluate fill-in-the-blank predictions

   ```
   python src/eval/eval_bert.py models_scratch/corpus_original_tiny
   ```

   Loads the trained model and predicts the best word for each [MASK] token in data/eval.txt.
   The output lists the top-5 predictions with their probabilities.

   For the filtered model:

   ```
   python src/eval/eval_bert.py models_scratch/corpus_filtered_tiny
   ```

4. Create next-word evaluation set

   ```
   python src/eval/make_nextword_eval.py data/corpus_original.txt --out_prompts data/nextword_eval.txt --out_answers data/nextword_answers.tsv --max_items 500
   ```

   Generates 500 short sentence fragments ending with [MASK], plus the correct next words.

   Key parts:

   * --out_prompts – output prompt file
   * --out_answers – file with correct next words
   * --max_items – number of examples to create

5. Score next-word prediction accuracy

   ```
   python src/eval/score_nextword_bert.py models_scratch/corpus_original_tiny data/nextword_eval.txt data/nextword_answers.tsv
   ```

   Evaluates how often the model predicts the correct next word.

   Outputs:

   * acc@1 – percent of correct top predictions
   * acc@5 – percent correct within top five guesses
   * avg_gold_prob – average model confidence in the correct answer

   For the filtered model:

   ```
   python src/eval/score_nextword_bert.py models_scratch/corpus_filtered_tiny data/nextword_eval.txt data/nextword_answers.tsv
   ```

---

Metrics Explained

acc@1 – The percentage of prompts where the model’s top prediction matches the correct next word.
acc@5 – The percentage where the correct word appears in the top five predictions.
avg_gold_prob – The average probability the model assigns to the correct next word across all prompts.

---

Evaluation Workflow Summary

1. Run make_corpora.py to create training corpora.
2. Train both models with train_bert_scratch.py.
3. Evaluate fill-in-the-blank predictions using eval_bert.py.
4. Generate and score next-word tasks using make_nextword_eval.py and score_nextword_bert.py.
5. Compare results between the original and filtered models.

---

Interpreting Results

These models are trained on very small datasets and are not expected to perform well in absolute accuracy.
The goal is not to achieve high scores but to compare how the filtered corpus affects the model’s behavior and generalization.
The model trained on corpus_original usually performs slightly better on prompts involving removed topics, confirming the influence of training data coverage on generalization.

---

Next Steps

1. Expand or rebalance the corpus to test larger or more selective filters.
2. Visualize vocabulary overlap and token distributions between corpora.
3. Add quantitative plots comparing acc@1 and acc@5 between models.
4. Experiment with different masking probabilities or architectures.
5. Document qualitative examples showing how predictions differ by corpus.

---

Maintainers: Mollie Hamman, Alex Geer, Erick Mainoo
Course: CS 5/7322 – Natural Language Processing, Fall 2025
Institution: Southern Methodist University (SMU), Dallas TX
Last Updated: October 2025

---

 short 1-paragraph “Abstract / Project Summary”
