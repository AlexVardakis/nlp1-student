# Project 1 – Sentiment Classification (brief)

Group 52, NLP-1 (UvA, block 2, 2026). Pulled from Alex's vault note _NLP-1 Project 1 - Sentiment Classification_ on 5 Nov 2026. Task file: `project.ipynb` in this folder.

## Key dates

| What                 | When                                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| Deliverable deadline | **Fri 6 Nov 2026, 23:59** (no penalty until Sun 8 Nov, 23:59)                                         |
| Oral assessment      | **Mon 9 Nov 2026, 15:00**, slot 1B, ~15 min with the TA, group grade. Be there 5 min early (late = 0) |
| Canvas               | [assignment 724688](https://canvas.uva.nl/courses/59944/assignments/724688)                           |

Grading context: the 3 practicals are 5% each of the course grade (exam 80%, blog post 5%). Submitting on time is what unlocks the oral.

## Task

Apply the Practical 1 tutorial (`tutorial.ipynb`: segmentation + text classification on AG News) to a **sentiment** dataset IMDB . The same task and dataset carry over to **projects 2 and 3 and the final blog post**, so experiments must stay clean and reproducible.

### Checklist (from `project.ipynb`)

1. **3×3 grid of classifiers**
   - Representations: token indicators, bag-of-words (tutorial) + **one of our own**.
   - Architectures: log-linear, one-hidden-layer MLP (tutorial) + **one of our own** (e.g. a modified MLP: more layers, other activation, dropout).
   - Tutorial systems are rerun on the new data; new ones are _added_, not substituted.
2. **Cross-validation** with a relevant metric. Accuracy is fine for balanced labels; with imbalance, macro-F1 is easier to defend. Justify the choice from the label distribution.
3. **Bayesian estimation with a ROPE** (Benavoli correlated t-test, `analysis.BenavoliCorrelatedT`) to decide whether one system is really better or practically equivalent. We choose and **motivate** the ROPE size (tutorial uses 0.005 = half an accuracy point).

### "Get to work!" topics (the TA asks about these; one backup slide each, max 6)

1. **Dataset**: size, examples, instances per label.
2. **Tokeniser**: reuse the tutorial BPE or train a new one? Hyperparameters (vocab size, min frequency)? Trained on what? The dataset may be too small, so maybe a larger corpus of similar text.
3. **Representations**: why our third one.
4. **Architectures + hyperparameters**: layers, activations, dropout.
5. **Evaluation**: was the method adequate for this setup? Anything else worth measuring?
6. **Analysis**: compare models, inspect errors, pick the best.

They can also ask about the **tutorial itself** (no deliverable; work through its questions).

## Stats reminders (from the Practical 1 notes)

- Don't just average fold accuracies: report the posterior and P(A wins) / P(Tie) / P(B wins).
- Folds share training data, so they are correlated; more repeats of the same CV can't shrink uncertainty below a floor (≈ σ²·ρ/(1−ρ), ρ ≈ 1/k).
- A high P(Tie) is real information. The ROPE is a judgement call about what difference matters, not something the statistics decides.
- Effect size = mean difference / SD; report it, it isn't a verdict.

## Rules

- Reusing the tutorial code and helpers (`clslib.py`, `experiment.py`, `analysis.py`, `loglin.py`, `smlp.py`, `bpe.py`) is explicitly allowed.
- Code isn't graded directly, but a reproducible version is shared with the blog post: keep it organised.
- **AI use is allowed but must be disclosed** in the blog post. Keep a log of what was delegated to assistants (see `AI_USE_LOG.md` if created, or note it in the notebook).

## Deliverables to have ready

- [ ] Completed `project.ipynb` (or a cleaned experiment notebook) with the 3×3 grid, CV results and ROPE analysis
- [ ] ≤ 6 backup slides, one per "Get to work!" topic
- [ ] Submission on Canvas by Fri 6 Nov 23:59
- [ ] AI-use record started
