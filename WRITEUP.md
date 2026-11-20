# A Self-Tuning Super Learner for Cold-Start Survey Completion

*SimulacraBench 2026, team atomiccninja. Code: https://github.com/bukhari-hamzamukhtar/SimulacraBench*

## Abstract

A held-out respondent arrives with only a short block of demographic answers, and every
other answer must be given as a probability distribution. We treat this as calibrated
conditional density estimation and solve it with a cross-validated Super Learner that fits
everything at run time on the visible rows: empirical-Bayes lookup tables, multinomial
logistic models, multitask and per-item neural networks, and nearest-neighbour hot-deck
donors, mixed per item by EM on out-of-fold predictions, then calibrated and adjusted for
the grader's probability floor. Two components are specific to survey data: a two-part
("hurdle") output for skip-logic items, and a hot-deck donor correction for respondents with
near-identical profiles. Because the real data are never released, we measure the method in
three synthetic worlds that each come with an exact or strong reference: the official
sandbox (where we reach 99% of the Bayes-optimal score), a structured stress test with an
exact oracle, and a two-round panel. No single model family wins in all three; the stack
does.

## 1. The problem, stated exactly

For a held-out respondent with GIVEN block x and item j with options 1..K_j (plus a
"never asked" sentinel for gated items), we report a vector q_j(x). The score is the mean of
log(eps + (1 - K eps) q_j(x)[y]) over hidden cells, with eps = 1e-3, reported as skill
relative to the crowd marginal. Held-out rows are an i.i.d. sample of the same population,
so the expected score is maximised by the conditional distribution P(y_j | x) item by item;
joint structure across a respondent's answers changes only the variance of the score. The
task is the first stage of multilevel regression and poststratification, or cold-start
completion in recommender terms.

## 2. Method

**Candidates.** Every candidate is fitted with 5-fold cross-validation, giving an honest
out-of-fold (OOF) prediction for every visible row; held-out rows get a full-data refit.
(i) The crowd marginal. (ii) *Lookup tables*: hierarchical Dirichlet backoff along the GIVEN
columns ranked by mutual information with the item, with the smoothing strength of each
level chosen by leave-one-out likelihood. (iii) *Logistic models*: multinomial regression
under Gaussian priors with main effects, with main effects restricted to each item's four
most informative columns (selection inside each fold), and with all pairwise interactions;
all items are fitted jointly with L-BFGS. (iv) *Multitask networks* with one shared trunk
over embeddings of the GIVEN block and one softmax per item, several widths and seeds.
(v) *Per-item networks*: shared embeddings followed by a hidden layer of its own for every
item, computed in one batched einsum. (vi) *Hot-deck donors*: the visible respondents within
Hamming distance 2 (or 4, with a gentler weight) of the respondent vote for their own
answers with weight exp(-lambda d), shrunk toward a lookup estimate. (vii) Ordinal-smoothed
twins and, when time allows, gradient-boosted trees.

**Two-part outputs for skip logic.** A gated item equals the sentinel when the respondent
was never asked. For gated items, lookup tables, logistic models and networks also come in
a hurdle form, P(y = k) = P(asked) P(k | asked) and P(sentinel) = 1 - P(asked), with separate
parameters for the two parts. The columns that decide whether a question is asked and the
columns that decide its answer then stop competing for the same capacity. Gates are learned,
not imposed, because the real gates hold only approximately.

**Stacking and calibration.** Per item, a convex mixture of the candidates is fitted to the
OOF predictions by EM (the log-likelihood is concave in the weights), with item weights
shrunk toward weights pooled over items by a strength chosen on a split of the OOF rows.
This is the Super Learner of van der Laan, Polley and Hubbard (2007); the floor bounds the
loss, which its oracle inequality requires. One temperature per item follows.

**Donor correction.** After stacking, near-identical donors (distance at most 2 or 4) pull
the stacked belief B toward their answers, P = (votes + alpha B) / (weight + alpha); rows
without a close donor keep B. The distance scale lambda, the cut-off and alpha are chosen
per item by OOF likelihood, and the correction is left off when it does not help. When
the GIVEN block has a survey-round column, a second set of donors is drawn from the other
round only. The UNHCR file holds one row per household per round, so a household
interviewed twice would have its other interview in the other round, while a same-round
lookalike with an identical profile is a different household and must not outvote it.
In our harder panel world a cut-off of 2 alone recovers almost none of the gain that a
cut-off of 4 does.

**Floor-aware reporting.** The grader mixes every vector with a flat one, so reporting the
belief itself is not optimal. Maximising the expected floored score gives a water-filling
transform, p_k = max(eps, c b_k), used per item when it scores better OOF.

**Budget.** A wall-clock governor gives each instrument a share of 45 minutes, carries
unused time forward, drops candidates whose projected cost does not fit, and lets any
candidate fail without failing the run. The full three-instrument run takes 18 minutes on an
L4 GPU (limit: 60 minutes on an H100), peak memory 5.0 GB (limit 16 GB).

## 3. Evaluation without the real data

We never see real answers, so we measure against references we can trust.

**Official sandbox.** One hidden normal trait per respondent drives every answer through a
softmax. Replaying the generator recovers its parameters, so the Bayes-optimal predictor is
computable exactly by integrating the trait posterior on a grid.

**Structured stress test.** The sandbox's GIVEN columns, but every answer is driven by them:
lookups on one or two columns, additive effects of two or three columns, thresholds on sums
of household counts, noise items, and the schema's skip logic on top (soft for UNHCR, as its
real gates are). Items are independent given x and their gate parent, so the oracle is exact.

**Two-round panel.** 40% of rows are second-round re-interviews of other households: the
GIVEN block is copied (round set to 2, each column changed with probability 0.05) and each
answer is carried over with probability 0.7. This mirrors the UNHCR file, which merges two
survey rounds fielded in the same two months; its excluded fields include consent to a
follow-up contact and an "interviewed before" question, so round 2 likely re-contacted
round-1 households. A near-duplicate version carries 95% of
answers over and gives 30% of first-round households the exact profile of another
household, since identical profiles are common in real survey data and rare in the
sandbox.

| world | system | UNHCR | World Bank | UNICEF |
|---|---|---|---|---|
| sandbox | ours | 0.2734 | 0.2767 | 0.3988 |
| sandbox | best possible | 0.2766 | 0.2803 | 0.4000 |
| stress | organisers' baseline | 0.2012 | 0.2977 | 0.5217 |
| stress | shared network only | 0.1952 | 0.2924 | 0.5272 |
| stress | logistic only | 0.2314 | 0.3083 | 0.5284 |
| stress | lookup tables only | 0.2560 | 0.3119 | 0.5270 |
| stress | ours | 0.2718 | 0.3253 | 0.5296 |
| stress | best possible | 0.2832 | 0.3338 | 0.5303 |
| panel | ours without donors | 0.2885 | 0.2460 | |
| panel | ours, without other-round donors | 0.3571 | 0.3094 | |
| panel | ours | 0.3685 | 0.3094 | |
| near-duplicate panel | ours, without other-round donors | 0.3345 | | |
| near-duplicate panel | ours | 0.3813 | | |

*Skill on held-out cells, seed 1, scored by the grader's own code; standard errors 0.002 to
0.010. On a second sandbox seed we score 0.2706 / 0.2962 / 0.3491 against a best possible
0.2737 / 0.2997 / 0.3494, so 99% of the ceiling again.*

The worlds pull in opposite directions. The shared-trunk network is the best single model on
the sandbox and the worst on the stress test, below the organisers' baseline; lookup tables
are the reverse; donors dominate the panel and do nothing elsewhere. Choosing per item lets
one method stay near the ceiling in all three.

## 4. Limitations

The synthetic worlds are informative about the method's adaptivity, not about the absolute
level of the real scores, which depend on how much the real GIVEN block reveals. What the
donor layer is worth depends on whether the real files contain the same household twice;
the UNHCR file carries no linking identifier, so we cannot check this, and where there are
no such twins the layer switches itself off. The stack's
weights are fitted on OOF predictions of the same rows they are applied to, which carries a
small optimism. We did not use language-model priors: on the evidence we reviewed
(Dominguez-Olmedo et al., NeurIPS 2024) their answer distributions are close to uniform
once ordering effects are removed, and the visible rows carry far more information.

## References

- M. J. van der Laan, E. C. Polley, A. E. Hubbard. Super Learner. *Statistical Applications
  in Genetics and Molecular Biology* 6(1), 2007.
- R. Dominguez-Olmedo, M. Hardt, C. Mendler-Duenner. Questioning the survey responses of
  large language models. *NeurIPS* 2024.
- R. J. A. Little, D. B. Rubin. *Statistical Analysis with Missing Data*, 3rd ed. Wiley, 2019
  (hot-deck imputation).
- J. G. Cragg. Some statistical models for limited dependent variables with application to
  the demand for durable goods. *Econometrica* 39(5), 1971 (two-part / hurdle models).
