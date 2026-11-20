# SimulacraBench: a self-tuning Super Learner for cold-start survey completion

Submission to the NeurIPS 2026 SimulacraBench competition (UNHCR, UNICEF and World Bank
survey instruments). The whole method lives in [`main.py`](main.py), which is the file that
was submitted. Everything else in this repository is there to measure it.

## The problem, stated exactly

A held-out respondent arrives with only their `GIVEN` block (7 to 17 demographic columns);
every one of their `PREDICT` answers is hidden. The score is the mean log probability over
all hidden cells, and held-out rows are an i.i.d. sample of the same population. So the
expected score is maximised by

    P(answer to item j | GIVEN block), one item at a time.

How a person's answers co-vary changes the variance of the score, not its expectation. The
task is calibrated conditional density estimation from categorical features: the first
stage of multilevel regression and poststratification in survey statistics, or "cold-start"
completion in recommender-system terms.

## The method

`predict()` fits everything at run time on the visible rows (the data is never released,
and the final phase allows one scored run).

1. **Candidates.** Each is fitted with 5-fold cross-validation, so every visible row gets
   an honest out-of-fold prediction. Held-out rows get a full-data refit (deterministic
   models) or a refit blended with the fold models (networks).
   - the crowd marginal (the zero of the skill scale);
   - **lookup tables**: empirical-Bayes hierarchical Dirichlet backoff over the GIVEN
     columns ranked by mutual information, smoothing strength chosen per item and per level
     by leave-one-out likelihood;
   - **logistic regression** (multinomial, Gaussian priors): main effects; main effects on
     each item's four most informative columns; main effects plus all pairwise
     interactions. All items are fitted at once with L-BFGS;
   - **multitask networks**: one shared trunk over embeddings of the GIVEN block, one
     softmax per item, several widths and seeds, early stopping on an inner split;
   - **per-item networks**: shared embeddings, then a hidden layer of its own for every
     item, computed in one batched einsum;
   - **hot-deck donors**: visible respondents whose GIVEN block differs in at most two
     (or, weighted down, four) columns vote for their own answers, adjusting a lookup
     estimate. This is the classic survey imputation method;
   - gradient-boosted trees when the time budget allows, and ordinal-smoothed twins of the
     strongest candidates for ordered answer scales.
2. **Two-part outputs for skip logic.** A gated item's answer is the gate sentinel when the
   respondent was never asked. For gated items the lookup tables, logistic models and
   networks each come in a second, two-part form:

       P(answer = k) = P(asked) * P(k | asked),   P(sentinel) = 1 - P(asked)

   (a hurdle model). The two parts have separate parameters, so the columns that decide
   whether a question is asked and the columns that decide its answer do not compete for
   the same capacity. Gates are learned from the data, never imposed, because the real
   data's gates hold only approximately.
3. **Stacking.** Per item, a convex mixture of the candidates is fitted to the out-of-fold
   predictions by EM (concave in the weights, so the optimum is global). Item weights are
   shrunk toward weights pooled over all items, with the strength chosen on a split of the
   out-of-fold rows. This is the Super Learner (van der Laan, Polley and Hubbard, 2007);
   the grader's probability floor bounds the loss, which is the condition its oracle
   inequality needs. The crowd marginal is always a candidate.
4. **Calibration.** One temperature per item, fitted on the same out-of-fold data.
5. **Donor correction.** After stacking, near-identical donors pull each respondent's
   stacked belief toward their own answers, `P = (votes + alpha * B) / (weight + alpha)`.
   A respondent without a close donor keeps `B` exactly. The distance scale, the cut-off
   and `alpha` are chosen per item by out-of-fold likelihood, and the correction is left
   off wherever it does not help. When the GIVEN block has a survey-round column (UNHCR's
   `wave`), a second set of donors is drawn from the other round only: the file holds one
   row per household per round, so a household interviewed in both rounds would have its
   other interview there, and a same-round lookalike with an identical profile must not
   outvote it. (`USE_DONORS` at the top of `main.py` turns donors off.)
6. **Floor-aware reporting.** The grader mixes every vector with a flat one before scoring,
   so reporting the belief itself is not optimal. Maximising the expected score of the
   floored vector gives a water-filling transform, used per item whenever it scores better
   out of fold.

A wall-clock governor keeps the run far inside the 60-minute limit, and any candidate can
fail without failing the submission.

## How it was measured

Three synthetic worlds. The first two come with an exact Bayes-optimal oracle, so every
score can be read against the best possible score rather than against another guess.

- **The official sandbox** (`public/make_sandbox.py`): one hidden trait per respondent
  drives every answer. Replaying the generator recovers its parameters exactly
  ([`oracle/main.py`](oracle/main.py)).
- **A structured stress test** ([`stress.py`](stress.py)): the sandbox's GIVEN columns, with
  every answer driven by them instead: lookups on one or two columns, additive effects,
  thresholds on sums of household counts, noise items, and the schema's skip logic on top
  (holding only approximately for UNHCR, as the real UNHCR gates do). Items are independent
  given the GIVEN block and their gate parent, so the oracle is exact.
- **A two-round panel** (`STRESS_MODE=panel`): the UNHCR file merges two survey rounds
  fielded in the same two months, and its excluded fields (consent to a follow-up contact,
  an "interviewed before" question in a post-return module) suggest that round 2
  re-contacted round-1 households. Here 40% of rows are second-round re-interviews: the
  GIVEN block is copied (round set to 2, each column changed with probability 0.05, or 0.2
  for a harder version with `PANEL_FLIP=0.2`) and each answer is carried over with
  probability 0.7. A near-duplicate version (`PANEL_COPY=0.95 PANEL_FLIP=0.02
  PANEL_LOOK=0.3`) carries 95% of answers over and gives 30% of first-round households
  the exact profile of another household, because identical profiles are common in real
  survey data and rare in the sandbox. There is no exact oracle; it measures whether a
  method uses near-identical donors when they exist.

The worlds pull in different directions. A shared neural trunk is the best single model in
the first and the worst in the second; lookup tables are the reverse; donors dominate the
third and switch themselves off in the other two. A method that does well in all three is
choosing per item.

## Results

Skill on held-out cells (0 = the crowd marginal, 1 = perfect), scored by the grader's own
code. Seed 1 unless stated. Standard errors are 0.002 to 0.010.

**The official sandbox**, against its exact best possible score:

| | UNHCR | World Bank | UNICEF | mean |
|---|---|---|---|---|
| this method, seed 1 | 0.2734 | 0.2767 | 0.3988 | 0.3163 |
| best possible, seed 1 | 0.2766 | 0.2803 | 0.4000 | 0.3190 |
| this method, seed 2 | 0.2706 | 0.2962 | 0.3491 | 0.3053 |
| best possible, seed 2 | 0.2737 | 0.2997 | 0.3494 | 0.3076 |

**The structured stress test**, against its exact best possible score:

| system | UNHCR | World Bank | UNICEF |
|---|---|---|---|
| organisers' baseline | 0.2012 | 0.2977 | 0.5217 |
| shared neural network only | 0.1952 | 0.2924 | 0.5272 |
| logistic regression only | 0.2314 | 0.3083 | 0.5284 |
| lookup tables only | 0.2560 | 0.3119 | 0.5270 |
| the stack before this test (no two-part outputs) | 0.2646 | 0.3212 | 0.5295 |
| **this method** | **0.2718** | **0.3253** | **0.5296** |
| best possible | 0.2832 | 0.3338 | 0.5303 |

**The two-round panel** (no exact ceiling):

| system | UNHCR | UNHCR, harder twins | UNHCR, near-duplicates | World Bank |
|---|---|---|---|---|
| lookup tables only | 0.2025 | | | |
| the stack without donors | 0.2885 | | | 0.2460 |
| donors from either round | 0.3571 | 0.2916 | 0.3345 | 0.3094 |
| **this method** (donors from the other round added) | **0.3685** | **0.3025** | **0.3813** | **0.3094** |

On the stress test the two-part outputs halve the gap on items behind skip logic, and the
UNHCR score moves from 93% to 96% of the best possible. On the panel, donors add about 0.08
on UNHCR and 0.06 on the World Bank. Drawing donors from the other round matters most when
same-round households share a profile: in the near-duplicate world a same-round lookalike
otherwise outvotes the household's own other interview. Where there are no twins (the
sandbox and the stress test) the donor layer switches itself off and the scores do not move.

**Cost.** All three instruments in one process, as hosted: 18 minutes on an L4 GPU (the
limit is 60 minutes on an H100), peak memory 5.0 GB (limit 16 GB).

## Reproducing

    pip install numpy pandas scipy scikit-learn torch pyyaml pyarrow
    git clone https://github.com/SituatedEvals/public
    python harness.py . unicef 2 1        # one instrument, scored the way the grader scores
    python run3.py . unhcr,world_bank,unicef 1   # all three in one process, as hosted
    python stress.py . unhcr 1            # the structured stress test
    python stress.py oracle unhcr 1       # its exact ceiling
    STRESS_MODE=panel python stress.py . unhcr 1                 # the two-round panel
    STRESS_MODE=panel PANEL_FLIP=0.2 python stress.py . unhcr 1  # harder twins
    STRESS_MODE=panel PANEL_COPY=0.95 PANEL_FLIP=0.02 PANEL_LOOK=0.3 python stress.py . unhcr 1  # near-duplicates
    ORACLE_SEED=1 python harness.py oracle unhcr 2 1   # the sandbox ceiling
    python robust.py .                    # adversarial inputs, every output checked legal

`harness.py`, `run3.py` and `stress.py` drive `predict()` through the grader's own
masking, validity checks, floor and skill formula, after the same JSON round trip the
hosted driver applies. A GPU is used when present; everything also runs on a CPU, slowly.

## Licence

MIT (see [`LICENSE`](LICENSE)), which permits non-commercial research reproduction.
