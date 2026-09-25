# Business Entity Resolution — Full Execution Pipeline (3-Day Plan, Team of 3)

**How to use this doc:** check off `- [ ]` items as you complete them. Every experiment has the same fixed spec (what/why/input/preprocessing/model/hyperparams/validation/cost/continue-criteria/abandon-criteria). Experiments within a phase are ordered as a ladder — each is expected to strictly beat the one before it, so you always have a checkpoint to fall back to if a later one underperforms. Numbers marked `[TBD from EDA]` depend on your actual dataset stats.

**Constraints baked into every phase:** no external API/DB lookups (fair play rule), final inference-path model must be MIT/Apache-2.0 and ≤8B params, output must pass `utils/validate_submission.py`, country is an open set (France appears only in test).

---

## Priority Legend

- **MUST DO** — required for a working, competitive submission.
- **SHOULD DO** — meaningful expected F_0.5 gain per hour invested.
- **IF TIME** — only after MUST + SHOULD are done with time left.
- **SKIP** — explicitly not worth it. Reasoning given, not just omitted.

---

## Team Split (3 people)

Phase 0 is the one phase that must be built **once, jointly** — not once per person. If each of you derives your own train/val split, none of your F_0.5 numbers will be comparable to each other, which is the single most common time-sink in team competitions. One person builds it, all three load the same output files.

After Phase 0, work splits into three tracks that run in parallel with light sync points:

| Track | Owns | Depends on |
|---|---|---|
| **A — Blocking & Data** | Phase 0 (builds it), Phase 1.1/1.2, Phase 2.1, Phase 3 | nothing upstream — starts first |
| **B — Classical Matching** | Phase 1.3, Phase 2.2–2.4, Phase 6 threshold/ensembling infra | Track A's blocking output |
| **C — Deep Learning / Fusion** | Phase 4, Phase 5 (all three fusion variants) | Track A's blocking output, Track B's engineered features (for the feature-tower) |

**Sync points:**
- End of Day 1 AM: Track A ships the canonical validation split + country-holdout fold + blocking-union candidates. Nothing downstream starts until this lands.
- End of Day 1: Track B's feature set (2.2–2.4) is frozen and handed to Track C to reuse as the feature-tower input for Phase 5.
- Phase 5 is the natural 3-way split point: **each person owns one fusion variant** (5.1 late / 5.2 gated / 5.3 cross-attention) in parallel, since all three share the same towers and only the fusion head differs. This is the best-parallelized phase in the whole plan — don't do it serially.
- Day 3 AM: all three converge on Phase 6 (ensembling needs every track's output).

---

## Day Map

| Day | Phase | Primary owner |
|---|---|---|
| Day 1 AM | Phase 0 | Track A (whole team reviews output before proceeding) |
| Day 1 PM | Phase 1 | Track A (blocking) + Track B (matcher) in parallel |
| Day 1 evening | Phase 2 | Track B |
| Day 2 AM | Phase 3 | Track A |
| Day 2 PM | Phase 4 | Track C |
| Day 2 evening – Day 3 AM | Phase 5 | Track A / B / C, one fusion variant each |
| Day 3 midday | Phase 6 | Whole team |
| Day 3 PM/evening | Phase 7 | Whole team (split: repro / methodology doc / final validation run) |

---

## Validation Strategy

Use all of these — they answer different questions, and none of them alone tells you what your private leaderboard score will actually be.

### 1. Entity-level random split
- **How:** split by `source1_entity_id`, never by pair (a pair-level split leaks — pairs from the same S1 entity would appear in both train and val)
- **Why:** fast iteration, matches the in-distribution (US+India) training data
- **What to expect:** the *highest* F_0.5 of any validation method here, because there's no distribution shift. Treat it as an upper bound for fast iteration, not as your real score estimate — France isn't represented at all.

### 2. Country-holdout validation (leave-one-country-out)
- **How:** train on US-only, evaluate on India-only pairs never seen in training; then repeat the reverse and average
- **Why:** your only proxy for the France shift in the real test set — nothing else in your validation setup simulates an unseen country
- **What to expect:** a real drop vs. the random split. A moderate drop is normal and expected. A *sharp* drop (roughly >0.15–0.2 absolute F_0.5 `[TBD — calibrate once you see your random-split number]`) means your features or model are implicitly overfitting to country-specific patterns (e.g. an address parser that assumes US-style formatting) — go audit for that before trusting the model on France.

### 3. Stratification by match-count / singleton status
- **How:** when building any split (random or country-holdout), preserve the ratio of singleton-vs-matched S1 entities from the full training set
- **Why:** F_0.5 macro-averages per entity, and singletons are worth a full point each — an unstratified split can accidentally over- or under-represent them, making your val score misleading
- **What to expect:** without stratification, val F_0.5 will swing noticeably between re-splits/reruns for no real modeling reason. With it, val score should move consistently in the direction your changes actually deserve.

### 4. Blocking recall@K (a separate metric from matcher validation)
- **How:** for each blocking method, measure the fraction of true matches that survive into the candidate set at your chosen K
- **Why:** this is the hard ceiling on the entire pipeline — no matcher can recover a match blocking never retrieved
- **What to expect:** recall should approach ~100% at a reasonably generous K for a healthy blocking method. If it plateaus below ~90%, that gap is a hard cap on your achievable recall no matter how good Phase 2–5 get — fix blocking first, every time, before tuning the matcher further.

### 5. K-fold / repeated stratified splits (for score stability across 3 parallel workstreams)
- **How:** run 3–5 stratified entity-level folds instead of a single split
- **Why:** with three people running experiments simultaneously and comparing numbers, a single noisy split makes small real improvements indistinguishable from noise
- **What to expect:** look at the spread across folds. If the fold-to-fold variance is larger than the gap between two experiments you're comparing, that "improvement" isn't trustworthy yet — don't lock in a decision on it.

### 6. Held-out threshold-tuning split (3-way split: train / model-selection val / threshold-tune)
- **How:** don't pick your best model *and* tune the F_0.5 decision threshold on the same validation set
- **Why:** doing both on one split quietly overfits the threshold to that split's specific noise
- **What to expect:** the threshold chosen on a fresh split should land close to the one chosen on your model-selection split. A big divergence between them is a sign you're overfitting the threshold — smooth it out (e.g. average the threshold across folds) rather than trusting a single sharp optimum.

### 7. Public vs. private leaderboard
- **What to expect:** the public leaderboard is scored on a subset of test; final rankings use the private portion. Some shake-up between public and private is normal, and it's likely larger here than in a typical comp because the country-shift risk (France) isn't something the public subset necessarily represents proportionally. Trust your own country-holdout validation more than small public-LB deltas — don't make late-game model choices chasing a public-LB gain that your local CV doesn't support.

### 8. Team validation hygiene
- One canonical output from Phase 0: an `entity_id → fold_id` mapping plus the country-holdout assignment, saved to a shared file, loaded identically by all three tracks.
- One shared scoring function (same label construction, same F_0.5 implementation) used by everyone — never let two people compute "F_0.5" slightly differently and compare numbers as if they're the same metric.

---

## Phase 0 — Foundation `[MUST DO]` — Track A builds, whole team reviews

- [ ] Load all 7 files, confirm columns match spec exactly
- [ ] Full EDA: row counts, null rates, duplicate rates per source
- [ ] `entity_id` prefix sanity check per file
- [ ] Country value distribution per source (watch for `US`/`USA`/`United States` inconsistency — already spotted: `"US"` vs `"India"`, not even the same representation style)
- [ ] Ground truth: match-count distribution, singleton fraction, S2-only vs S3-only vs mixed composition
- [ ] Check whether Source 1 is genuinely deduplicated internally
- [ ] Leakage check: no `entity_id` overlap between train and test; no row-order/index correlation with match likelihood
- [ ] **Ship the canonical validation split** (entity-level, stratified by singleton status) as a shared file
- [ ] **Ship the country-holdout fold** (US-only train / India-only val, and reverse) as a shared file
- [ ] **Ship the shared F_0.5 scoring function** everyone imports

**Abandon criteria:** none — always runs, nothing here gets cut for time.

---

## Phase 1 — Rule-Based Baseline `[MUST DO]` — Track A (blocking) + Track B (matcher)

Goal: prove the full harness works — blocking → matching → `matching_results.tsv` + `candidate_pairs.tsv` → `validate_submission.py` → leaderboard — before investing in better models.

### Experiment 1.1 — Exact-match blocking + exact-match matcher (dumbest possible pipeline)
- **Trying:** normalize name (lowercase, strip punctuation/legal suffixes), block + match only on exact normalized-name equality within the same normalized country
- **Why:** the fastest possible way to prove the harness end-to-end works before any real modeling
- **Cost:** CPU, minutes
- **Continue if:** output passes `validate_submission.py` and returns `SCORED` — that's the actual pass condition, not the score
- **Abandon:** never — always the very first checkpoint, move on regardless of score

### Experiment 1.2 — TF-IDF blocking (beats 1.1 on recall)
- **Trying:** TF-IDF (char 3–5 gram) over normalized `name + address`, cosine similarity, top-K = 20 per S1 entity
- **Why:** exact match in 1.1 misses almost every real variant — this recovers candidates 1.1 structurally can't
- **Validation:** blocking recall@K (Validation Technique 4) — should clearly beat 1.1's recall
- **Continue if:** recall ≥ ~85–90% `[TBD]`
- **Abandon/fix if:** recall low → raise K or add a second blocking key before Phase 2

### Experiment 1.3 — Cosine-threshold matcher on top of 1.2 (beats 1.1's exact-match matcher on precision)
- **Trying:** keep a candidate if cosine similarity > threshold T, T grid-searched on validation to maximize macro F_0.5
- **Why:** first real precision-tuned decision rule, still model-free
- **Continue if:** clearly beats 1.1's exact-match matcher on val F_0.5
- **Abandon:** N/A, always ships — Phase 2 starts regardless

---

## Phase 2 — Feature-Engineered Classical ML Matcher `[MUST DO]` — Track B

### Experiment 2.1 — Blocking union (beats 1.2 on recall)
- **Trying:** union of TF-IDF top-K + exact normalized-name-token block + exact-country block
- **Why:** different blocking methods fail on different noise types; union raises the recall ceiling
- **Continue if:** recall improves over 1.2
- **Abandon if:** union barely moves recall — 1.2 was already near-ceiling, don't over-invest further here

### Experiment 2.2 — Logistic regression on a small feature set (beats 1.3 on precision)
- **Trying:** logistic regression over 4–5 basic features (name Levenshtein ratio, name token Jaccard, address token Jaccard, TF-IDF cosine, country-match flag)
- **Why:** first real learned matcher — fast to train, easy to debug, establishes whether *any* ML model beats the rule-based threshold before adding complexity
- **Cost:** CPU, seconds
- **Continue if:** beats 1.3 on val F_0.5 — confirms the feature/label pipeline is sound
- **Abandon if:** underperforms 1.3 → that's a bug (label construction or feature bug), not a modeling problem — fix before adding GBM complexity on top of a broken foundation

### Experiment 2.3 — Full-feature LightGBM (beats 2.2)
- **Trying:** LightGBM binary classifier on the full engineered feature set
- **Input/features:** name — Levenshtein ratio, Jaro-Winkler, token-set-ratio, token-sort-ratio (`rapidfuzz`), token Jaccard, TF-IDF cosine (word + char), length delta, legal-suffix-normalized exact-match flag; address — token Jaccard, edit-distance ratio, shared-numeric-token flag, normalized-country-match flag (feature, not a hard filter); cross — name_sim × address_sim interaction, min/max/mean of the two
- **Hyperparameters:** `num_leaves` ~31, `learning_rate` ~0.05, `scale_pos_weight` tuned for class imbalance
- **Validation:** entity-level split (Technique 1) + K-fold (Technique 5) to confirm the gain over 2.2 is real, not noise
- **Continue if:** clear jump over 2.2 (e.g. +0.1 F_0.5 `[TBD]`), feature importances dominated by name/address sim as expected
- **Abandon if:** negligible gain over 2.2 → suspicious, re-check for leakage before assuming GBM "just doesn't help here"

### Experiment 2.4 — 2.3 + F_0.5-tuned threshold on a held-out threshold split (beats 2.3)
- **Trying:** re-tune the decision threshold specifically to maximize macro F_0.5, using Validation Technique 6 (separate split from the one used to pick the 2.3 model)
- **Why:** the default 0.5 cutoff is never optimal for a precision-weighted metric; doing this on a fresh split (not the model-selection split) avoids overfitting the threshold
- **Continue if:** beats 2.3's score with the default/naively-tuned threshold
- **Abandon:** never — this step is nearly free given 2.3 already exists, always do it

---

## Phase 3 — Embedding-Based Semantic Blocking `[SHOULD DO]` — Track A

### Experiment 3.1 — Off-the-shelf bi-encoder blocking (beats 2.1 on recall for semantic-variant cases)
- **Trying:** `all-MiniLM-L6-v2` (Apache-2.0) or `BAAI/bge-small-en-v1.5` (MIT), embed normalized name(+address), nearest-neighbor retrieval, union with 2.1
- **Why:** catches transliteration/word-order/abbreviation variants that lexical methods miss
- **Continue if:** candidate recall measurably improves over 2.1
- **Abandon/deprioritize if:** improvement is marginal (<2–3%) — keep classical blocking primary, redirect time to the matcher

### Experiment 3.2 — Contrastively fine-tuned bi-encoder `[IF TIME]` (beats 3.1 if 3.1 clearly helped)
- **Trying:** fine-tune the 3.1 encoder on your labeled training pairs (contrastive/triplet loss: true matches pulled together, hard negatives pushed apart) instead of using it off-the-shelf
- **Why:** specializes the embedding space to this exact task's noise patterns rather than generic sentence similarity
- **Cost:** GPU, low tens of minutes
- **Continue if:** recall or downstream matcher score improves over 3.1
- **Abandon if:** 3.1 itself didn't clearly help — don't invest further in this line

---

## Phase 4 — Fine-Tuned Transformer Pairwise Matcher `[SHOULD DO / IF TIME]` — Track C

### Experiment 4.1 — Cross-encoder fine-tune (beats 2.4 on cases needing cross-field token interaction)
- **Trying:** Ditto-style serialization — `[NAME] name1 [ADDR] addr1 [CTRY] country1 [SEP] [NAME] name2 [ADDR] addr2 [CTRY] country2` — fine-tune DistilBERT-base or MiniLM as a binary classifier
- **Why:** captures token-level cross-field interactions (e.g. "Corp" ≈ "Corporation") that hand-crafted features can miss; also becomes the reusable encoder for Phase 5's towers
- **Hyperparameters:** batch size 16–32, LR ~2e-5, 2–4 epochs, early stop on val F_0.5
- **Cost:** GPU, 30 min – 2 hr depending on `[TBD from EDA]` dataset size
- **Continue if:** beats 2.4 on val F_0.5
- **Abandon if:** unstable/overfits, or 2.4 already strong — fold its output probability into Phase 6's ensemble as one more feature rather than scrapping it

### Experiment 4.2 — Hard-negative-mined retrain `[IF TIME]` (beats 4.1 if 4.1 succeeded)
- **Trying:** use 4.1 to mine its own false positives from the blocking candidates, add them as hard negatives, retrain
- **Why:** classic ER improvement — the first training pass's blind spots become the second pass's explicit training signal
- **Continue if:** meaningful precision gain over 4.1, since these are exactly the near-miss cases F_0.5 punishes hardest
- **Abandon if:** 4.1 wasn't reached or time is tight — this is a refinement on an already-working model, not a foundation

---

## Phase 5 — Custom Multi-Field Fusion Architecture `[MUST DO — the stated deliverable]` — split 1 variant per person

Design only after Phase 2/4 error analysis shows which fields actually carry signal. "Multimodal" here means multi-field fusion (name/address/country as separate text views) — no image/audio exists in this data.

**Shared architecture (build once, all three reuse):**
- Name tower: MiniLM/BGE-small → name embedding
- Address tower: same encoder family → address embedding
- Engineered-feature tower: Phase 2.3's features → small MLP → feature embedding
- Country: learned embedding with an explicit "unknown" bucket (required for France)

### Experiment 5.1 — Late fusion `[MUST DO]` (the fusion-phase baseline everything else must beat)
- Concatenate tower outputs, small MLP/logistic regression on top
- Cheapest, fastest — implement this first regardless of who's assigned what else
- Cost: minutes on top of already-computed tower outputs

### Experiment 5.2 — Gated fusion `[MUST DO]` (targets beating 5.1 specifically on sparse-address cases)
- Learn per-field gates that scale each tower's contribution per example
- Directly motivated by this dataset's "missing address components" noise pattern — lets the model down-weight a noisy/incomplete field rather than trusting it equally every time
- **Continue if:** beats 5.1 specifically on the subset of validation entities with sparse/short addresses — check this subgroup explicitly, not just the overall score

### Experiment 5.3 — Cross-attention fusion `[IF TIME]` (most expressive, only if 5.1/5.2 are done)
- Name tower attends over address tower tokens before pooling
- Highest implementation and compute cost of the three — only attempt with time remaining
- Cost: GPU

**Continue/abandon per variant:** keep whichever beats Phase 4.1's single-encoder baseline on val F_0.5. If none beat it, that's a real, reportable finding (concatenated features already capture what's there) — say so honestly in the methodology doc rather than forcing a marginal win.

---

## Phase 6 — Ensembling, Thresholding, Error Analysis `[MUST DO]` — whole team

### Experiment 6.1 — Simple averaging ensemble (beats the single best model from 2.4 / 4.1 / 5.x)
- Average probabilities from GBM (2.3/2.4) + transformer (4.1) + best fusion model (5.x)
- **Continue if:** beats the best single model on val F_0.5

### Experiment 6.2 — Stacked/logistic blending ensemble (beats 6.1)
- Train a small logistic regression on top of the three models' output probabilities instead of a flat average
- **Continue if:** beats 6.1 — usually does, cheap to try, don't skip it

### Experiment 6.3 — Per-source calibrated thresholds (beats 6.2 if S2/S3 matching quality differs)
- **Trying:** separate decision thresholds for S1–S2 candidates vs. S1–S3 candidates, tuned independently on validation
- **Why:** the two sources may carry different noise profiles; a single global threshold can be suboptimal for one of them
- **Continue if:** per-source thresholds beat a single global threshold in 6.2
- **Abandon if:** S2 and S3 candidate quality look statistically similar in EDA — not worth the added complexity

- [ ] Error analysis: inspect false positives first (F_0.5 punishes these harder)
- [ ] Re-check performance on the **country-holdout fold** specifically before finalizing threshold
- [ ] Confirm true singletons in validation are still predicted empty at the chosen threshold

---

## Phase 7 — Final Packaging `[MUST DO]` — split 3 ways

- [ ] **Person 1:** run `utils/validate_submission.py`, fix every flagged issue, confirm final inference-path model(s) are MIT/Apache-2.0 and ≤8B params
- [ ] **Person 2:** `README.md` with exact reproduction steps, pinned `requirements.txt`, fixed random seeds
- [ ] **Person 3:** fill in `Documentation_template.md` — the Phase 5 fusion comparison (including any negative result) is your ablation section, already done
- [ ] Whole team: verify zip structure matches the required layout, submit with real buffer before the deadline

---

## Explicitly Skipped `[SKIP]`

- **Full LLM (Llama-3/Qwen 3B+) as primary matcher** — overkill for short-text pairwise classification given the time budget
- **Collective/graph-based ER** (joint clustering across all pairs) — too time-expensive to implement correctly in 3 days
- **Heavy hyperparameter search (Optuna etc.)** — sane defaults + light manual tuning only
- **Any external geocoding/business-registry lookups** — banned by the fair-play rule regardless of value
- **Per-person idiosyncratic validation splits** — covered above, but worth repeating as an anti-pattern: never let each team member build their own split

---

## Experiment Tracker

| Experiment | Owner | Features | Model | Val F_0.5 | Train Time | Result | Next Action |
|---|---|---|---|---|---|---|---|
| 1.1 Exact-match pipeline | A | normalized name+country | rule | | | | |
| 1.2 TF-IDF blocking | A | char n-gram | — | recall: __ | | | |
| 1.3 Threshold matcher | B | cosine sim | threshold | | | | |
| 2.1 Blocking union | A | multi-key | — | recall: __ | | | |
| 2.2 Logistic regression | B | 5 basic features | LogReg | | | | |
| 2.3 Full-feature GBM | B | engineered sim features | LightGBM | | | | |
| 2.4 GBM + tuned threshold | B | same + threshold split | LightGBM | | | | |
| 3.1 Embedding blocking | A | MiniLM/BGE embeddings | — | recall: __ | | | |
| 3.2 Fine-tuned bi-encoder | A | contrastive embeddings | — | recall: __ | | | |
| 4.1 Cross-encoder matcher | C | serialized pair text | DistilBERT/MiniLM | | | | |
| 4.2 Hard-negative retrain | C | + mined negatives | DistilBERT/MiniLM | | | | |
| 5.1 Late fusion | A/B/C | tower outputs | MLP | | | | |
| 5.2 Gated fusion | A/B/C | tower outputs | gated MLP | | | | |
| 5.3 Cross-attention | A/B/C | tower tokens | cross-attn | | | | |
| 6.1 Averaging ensemble | team | blended probs | avg | | | | |
| 6.2 Stacked ensemble | team | blended probs | LogReg | | | | |
| 6.3 Per-source thresholds | team | source-split thresholds | — | | | | |

---

## Still Blocking Progress

This doc is built from the problem statement and the sample rows shared so far — parameters marked `[TBD from EDA]` need real dataset numbers before Phase 1 starts. Run the diagnostic script from earlier (or upload the actual TSVs) so Phase 0's checkboxes get real numbers.
