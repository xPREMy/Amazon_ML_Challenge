# Business Entity Resolution — Full Execution Pipeline (3-Day Plan)

**How to use this doc:** check off `- [ ]` items as you complete them. Every experiment has the same fixed spec (what/why/input/preprocessing/model/hyperparams/training/validation/cost/continue/abandon) so you can execute without re-deriving anything mid-competition. Numbers marked `[TBD from EDA]` depend on your actual dataset stats — plug them in once you run the diagnostic script and stop treating this as final until you do.

**Constraints baked into every phase below:** no external API/DB lookups (fair play rule), final inference-path model must be MIT/Apache-2.0 and ≤8B params, output must pass `utils/validate_submission.py`, country is an open set (France appears only in test).

---

## Priority Legend

- **MUST DO** — required for a working, competitive submission. Do not skip.
- **SHOULD DO** — meaningful expected F_0.5 gain per hour invested. Do unless Day 1/2 ran long.
- **IF TIME** — do only if MUST + SHOULD are done with time left before Day 3 evening.
- **SKIP** — explicitly not worth it for this task/timeline. Reasoning given, not just omitted.

## Day Map

| Day | Phases | Goal |
|---|---|---|
| Day 1 AM | Phase 0 | EDA, leakage check, validation split design |
| Day 1 PM | Phase 1 | Rule-based baseline, first scored leaderboard submission |
| Day 1 evening | Phase 2 | Feature-engineered GBM matcher |
| Day 2 AM | Phase 3 | Embedding-based blocking |
| Day 2 PM | Phase 4 | Fine-tuned transformer pairwise matcher |
| Day 2 evening – Day 3 AM | Phase 5 | Custom multi-field fusion architecture |
| Day 3 midday | Phase 6 | Ensembling, threshold tuning, error analysis |
| Day 3 PM/evening | Phase 7 | Final packaging, validation, submission (leave buffer) |

---

## Phase 0 — Foundation (no modeling) `[MUST DO]`

- [ ] Load all 7 files, confirm columns match spec exactly
- [ ] Full EDA: row counts, null rates, duplicate rates per source
- [ ] `entity_id` prefix sanity check per file (no cross-contamination)
- [ ] Country value distribution per source (watch for `US`/`USA`/`United States` style inconsistency — already spotted in your sample: `"US"` vs `"India"`, not even the same representation style)
- [ ] Ground truth: match-count distribution, singleton fraction, S2-only vs S3-only vs mixed match composition
- [ ] Check whether Source 1 is genuinely deduplicated (near-duplicate names within S1 itself)
- [ ] Leakage check: no `entity_id` overlap between train and test; no row-order/index correlation with match likelihood
- [ ] **Validation split** — split by S1 entity (never by pair, or you leak), stratified to preserve singleton fraction
- [ ] **Country-holdout fold** — train blocking/matching using only US, validate on India-only (and vice versa) as a proxy for France generalization. This is your only signal on the open-set-country risk before the real test set tells you.

**Abandon criteria:** none — this phase always runs, it's not optional and nothing here gets cut for time.

---

## Phase 1 — Rule-Based Baseline `[MUST DO]`

Goal: get *something* through the entire pipeline — blocking → matching → `matching_results.tsv` + `candidate_pairs.tsv` → `validate_submission.py` → leaderboard — on Day 1. Competition hygiene: confirm the harness works before investing in better models, not after.

### Experiment 1.1 — Lexical blocking
- **Trying:** TF-IDF (character 3–5 gram) blocking over `normalized_name + address`
- **Why:** cheapest possible recall-focused candidate generator; sanity-checks the whole pipeline fast
- **Input:** `business_name`, `business_address` (concatenated)
- **Preprocessing:** lowercase, strip punctuation, strip common legal suffixes (`inc`, `corp`, `ltd`, `pvt`, `llc`, `co`, `limited`, `private`) via a small suffix list; collapse whitespace
- **Model:** `sklearn.TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))` + cosine similarity (brute-force `NearestNeighbors` or sparse matmul if dataset is small; approximate if `[TBD from EDA]` row counts push into six figures)
- **Hyperparameters:** top-K = 20 per S1 entity (revisit after measuring recall ceiling)
- **Training:** none — this is retrieval, not learning
- **Validation:** measure **candidate recall** = fraction of true matches present in the candidate set, on your held-out val split
- **Cost:** CPU, minutes
- **Continue if:** candidate recall ≥ ~85–90% `[TBD — tune target once you see singleton fraction and typical noise level]`
- **Abandon/fix if:** recall is low → raise K, or union with a second blocking key (e.g., shared first name-token) before moving to Phase 2

### Experiment 1.2 — Threshold matcher
- **Trying:** keep a candidate as a match if TF-IDF cosine similarity > threshold T
- **Why:** dumbest possible matcher, but it's a real submission
- **Input:** cosine similarity score from 1.1
- **Model:** none — a single tuned scalar threshold
- **Training:** grid-search T on validation set, maximize macro F_0.5 (not accuracy, not F1)
- **Cost:** seconds
- **Continue if:** validator passes cleanly and you get a `SCORED` status on the leaderboard — this is the real pass condition for Phase 1, not the score itself
- **Abandon:** N/A, always ships — move to Phase 2 regardless of score, this phase's job was proving the harness works

---

## Phase 2 — Feature-Engineered Classical ML Matcher `[MUST DO]`

### Experiment 2.1 — Wider blocking (union of methods)
- **Trying:** union of TF-IDF top-K (1.1) + exact normalized-name-token block + exact-country block
- **Why:** different blocking methods catch different failure modes; union raises the recall ceiling that every downstream model is capped by
- **Cost:** CPU, minutes
- **Continue if:** recall ceiling improves over 1.1
- **Abandon if:** union barely moves recall — Phase 1's blocking was already near-ceiling, don't over-engineer this further, spend time on the matcher instead

### Experiment 2.2 — Gradient-boosted pairwise classifier
- **Trying:** LightGBM/XGBoost binary classifier over engineered similarity features
- **Why:** this is where most of your competitive ER score usually comes from — classical feature engineering is historically very strong on structured/short-text ER, often on par with or beating deep models
- **Input/features per (S1, candidate) pair:**
  - Name: Levenshtein ratio, Jaro-Winkler, token-set-ratio, token-sort-ratio (use `rapidfuzz`), token Jaccard, TF-IDF cosine (word + char n-gram), length delta, legal-suffix-normalized exact-match flag
  - Address: token Jaccard, edit-distance ratio, shared-numeric-token flag (street numbers), normalized-country-match flag (feature, not a hard filter — don't exclude candidates on country mismatch, the string-representation inconsistency you already found means a real match could disagree on country string)
  - Cross: name_sim × address_sim interaction, min/max/mean of the two
- **Preprocessing:** same normalization as 1.1, applied consistently across all three sources
- **Model:** LightGBM (`num_leaves` ~31, `learning_rate` ~0.05, `scale_pos_weight` tuned for the heavy class imbalance you'll have post-blocking — positives are rare vs. all candidate pairs)
- **Training:** positives = ground-truth pairs recovered by your blocking; negatives = blocked-but-non-matching candidates (hard negatives) + a smaller sample of random negatives for long-tail coverage
- **Validation:** entity-level split from Phase 0 (never pair-level — a pair from the same S1 entity in both train and val leaks)
- **Threshold:** grid-search on val to directly maximize macro F_0.5, not the 0.5 default
- **Cost:** CPU, minutes to low tens of minutes
- **Continue if:** clear jump over Phase 1 baseline (e.g. +0.1 F_0.5 or more `[TBD]`), feature importances make sense (name/address sim should dominate)
- **Abandon if:** negligible improvement or worse than baseline — that's a label-construction or leakage bug, debug before touching embeddings

---

## Phase 3 — Embedding-Based Semantic Blocking `[SHOULD DO]`

- **Trying:** bi-encoder embedding blocking — `all-MiniLM-L6-v2` (Apache-2.0) or `BAAI/bge-small-en-v1.5` (MIT) over normalized name (+ address), brute-force or FAISS nearest-neighbor retrieval, unioned with Phase 2.1's candidates
- **Why:** catches semantic variants lexical methods miss — transliteration, word-order transposition, abbreviation expansion without shared character n-grams. (This is the same embed-and-retrieve pattern as your C.A.R.E. project's retrieval stage — same mechanics, different domain.)
- **Preprocessing:** same name/address normalization; encode once per source, cache embeddings
- **Hyperparameters:** top-K = 20–30, cosine similarity retrieval
- **Cost:** CPU-feasible for MiniLM at this scale; GPU speeds it up but isn't required
- **Continue if:** candidate recall measurably improves over Phase 2.1
- **Abandon/deprioritize if:** improvement is marginal (<2–3%) — many ER problems are already near lexical-blocking ceiling, and this is a precision-bound competition, so extra recall isn't worth much once you're already capturing most true matches. Keep classical blocking as primary, spend remaining time on the matcher.

---

## Phase 4 — Fine-Tuned Transformer Pairwise Matcher `[SHOULD DO / IF TIME]`

- **Trying:** Ditto-style approach — serialize each pair as one sequence: `[NAME] name1 [ADDR] addr1 [CTRY] country1 [SEP] [NAME] name2 [ADDR] addr2 [CTRY] country2`, fine-tune a small pretrained encoder (DistilBERT-base or MiniLM) as a binary sequence classifier
- **Why:** captures token-level cross-field interactions (e.g. "Corp" ≈ "Corporation") without hand-listing every synonym; also doubles as a reusable encoder for Phase 5's towers
- **Input:** same labeled pairs dataset as Phase 2.2
- **Hyperparameters:** batch size 16–32, learning rate ~2e-5, 2–4 epochs, early stopping on val F_0.5
- **Training:** needs GPU (Colab/Kaggle free tier is enough); budget 30 min – 2 hr depending on `[TBD from EDA]` dataset size
- **Validation:** same entity-level split
- **Continue if:** beats Phase 2.2's GBM on val F_0.5
- **Abandon if:** unstable/overfits with limited data, or GBM already very strong — don't scrap the effort, fold its output probability in as one more feature into a stacked/blended final model (Phase 6) instead of a standalone model

---

## Phase 5 — Custom Multi-Field Fusion Architecture `[MUST DO — this is the stated goal]`

Design only *after* Phase 2/4 error analysis tells you which fields actually carry signal — don't design blind. Reframe "multimodal" as **multi-field fusion**: there's no image/audio here, just name/address/country as separate text views that need combining (this mirrors real ER systems like DeepMatcher/Ditto/HierMatcher).

**Architecture:**
- Name tower: small encoder (MiniLM/BGE-small) → name embedding
- Address tower: same encoder family → address embedding
- Engineered-feature tower: Phase 2.2's similarity features → small MLP → feature embedding
- Country handling: learned embedding with an explicit "unknown" bucket (required — test has France, unseen in training)

Given the 3-day budget, compare **2–3** fusion strategies, not all 7 on your original list:

### Experiment 5.1 — Late fusion `[MUST DO]`
- Concatenate each tower's output (or scalar prediction), small MLP/logistic regression on top
- Cheapest, fastest — this is the fusion-phase baseline everything else must beat
- Cost: minutes on top of already-computed tower outputs

### Experiment 5.2 — Gated fusion `[MUST DO]`
- Learn per-field gates that scale each tower's contribution based on input
- Directly motivated by this dataset: address is sometimes sparse/missing components (per the problem statement's noise patterns) — gates let the model down-weight a noisy/incomplete field per-example rather than always trusting it equally
- Cost: low — small additional gating network, same tower embeddings reused

### Experiment 5.3 — Cross-attention fusion `[IF TIME]`
- Name tower attends over address tower tokens (or vice versa) before pooling
- Most expressive, most compute and implementation time — only attempt if 5.1/5.2 are done and time remains
- Cost: highest of the three, needs GPU

**Continue/abandon per variant:** keep whichever beats the Phase 4 single-encoder baseline on val F_0.5. If none beat it — that's a real, reportable finding (concatenated features already capture what's there; full fusion complexity isn't buying anything on this data), not a failure. Say so honestly in the methodology doc rather than forcing a marginal win.

---

## Phase 6 — Ensembling, Thresholding, Error Analysis `[MUST DO]`

- [ ] Blend GBM (2.2) + Transformer (4) + best fusion model (5) probabilities — simple average or logistic stacking
- [ ] Re-tune final decision threshold on val to maximize macro F_0.5 (ensembling shifts the optimal cutoff)
- [ ] Error analysis: inspect false positives first (F_0.5 punishes these harder than false negatives) — near-miss non-matches your model is over-trusting
- [ ] Re-check performance on the **country-holdout fold** specifically before finalizing threshold — don't tune only on the easy in-distribution split when test has an entirely unseen country
- [ ] Confirm every S1 validation entity that's a true singleton is still being predicted empty at your chosen threshold (this is where careless thresholds bleed the most macro-average points)

---

## Phase 7 — Final Packaging `[MUST DO]`

- [ ] Run `utils/validate_submission.py`, fix every flagged issue
- [ ] Confirm the final inference-path model(s) are MIT/Apache-2.0, ≤8B params — check every component actually used at inference, not just the ones you experimented with
- [ ] `README.md` with exact reproduction steps (data → blocking → matching → output), pinned `requirements.txt`, fixed random seeds
- [ ] Fill in `Documentation_template.md` — your Phase 5 fusion comparison (including any negative result) is your ablation section, already done
- [ ] Verify zip structure matches the required layout exactly
- [ ] Submit with real buffer before the deadline — not at the last minute

---

## Explicitly Skipped `[SKIP]`

- **Full LLM (Llama-3/Qwen 3B+) as primary matcher** — overkill for short-text pairwise classification; fine-tuning/inference cost isn't justified by expected gain over a small encoder within 3 days
- **Collective/graph-based ER** (joint clustering across all pairs) — theoretically elegant, too time-expensive to implement correctly in this window, and the fixed-reference (S1) structure of this task doesn't clearly need it
- **Heavy hyperparameter search (Optuna etc.)** — sane defaults + light manual tuning only; sweep budget isn't worth it against modeling/EDA time
- **Any external geocoding/business-registry lookups** — banned by the fair-play rule regardless of expected value

---

## Experiment Tracker

| Experiment | Features | Model | Val F_0.5 | Train Time | Result | Next Action |
|---|---|---|---|---|---|---|
| 1.1 Lexical blocking | TF-IDF char n-gram | — | recall: __ | | | |
| 1.2 Threshold matcher | cosine sim | threshold | | | | |
| 2.1 Blocking union | multi-key | — | recall: __ | | | |
| 2.2 GBM matcher | engineered sim features | LightGBM | | | | |
| 3 Embedding blocking | MiniLM/BGE embeddings | — | recall: __ | | | |
| 4 Transformer matcher | serialized pair text | DistilBERT/MiniLM | | | | |
| 5.1 Late fusion | tower outputs | MLP | | | | |
| 5.2 Gated fusion | tower outputs | gated MLP | | | | |
| 5.3 Cross-attention | tower tokens | cross-attn | | | | |
| 6 Ensemble | blended probs | avg/stacked | | | | |

---

## Still Blocking Progress

This doc is built from the problem statement and the sample rows you've shared — several parameters above are marked `[TBD from EDA]` because they genuinely depend on real dataset size, singleton fraction, and noise level. Run the diagnostic script from earlier (or upload the actual TSVs) so Phase 0's checkboxes can get real numbers before Phase 1 starts.
