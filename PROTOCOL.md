# Public-data lifecycle-routing pilot: frozen before outcome analysis

26 October 2026. The broad payback framing is already present in SaveRouter (arXiv:2609.37402); it is NOT claimed as novel. This pilot investigates decision uncertainty: how often point-estimated quality and amortized savings give misleading deployment recommendations, and whether an independent joint assessment identifies those cases. A confidence-bound wrapper alone is not method novelty.

## Public source and scope

Official withmartian/routerbench zero-shot matrix, repository revision 784021482c3f320c6619ed4b3bb3b41a21424fcb, SHA256 ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d. Metadata inspection found 36,497 cached queries, 11 models, scores and historical per-response costs. No paid LLM calls are made. Historical costs are neither current prices nor measured serving latency. Source license is not specified in the downloaded metadata; raw data are not redistributed in the code archive.

Choose workloads by metadata size and task type, before reading aggregate quality: mathematics (grade-school-math), commonsense/science reasoning (hellaswag, winogrande, arc-challenge), and professional law (mmlu-professional-law). These tasks use correctness-style scores; assert binary scores on all accessed development observations. Other tasks, including open-ended judge scores, are outside this pilot.

Normalize prompt whitespace and group exact duplicate prompts. Stable SHA256 partitions: 30% train, 10% hyperparameter/policy tuning, 20% independent calibration, 20% pilot evaluation, 10% fresh confirmation and 10% final. The first schema-inspected prompt is explicitly assigned to training and excluded from both held-out pools. Entire duplicate groups receive the same assignment. Prompt text alone and metadata determine splits. No final or fresh score/cost rows are used in training, tuning, calibration or pilot evaluation. Near duplicates and model-pretraining contamination are unresolved limitations of cached public data.

## Models and fair tuning

Reference is the predetermined GPT-4-1106-preview column. Candidate pool includes all 11 recorded models, including fixed alternatives; expensive routing is compared with useful fixed-model alternatives, not only a weak random router.

For each workload, three training-subset seeds (3101,3102,3103) and budgets 256/1024 queries (capped at available training groups). Same training rows and paid feedback matrix for all learned families. TF-IDF vocabulary fit on train only, at most 12,000 word uni/bigrams. Ridge score prediction: alpha={1,10,100}. Cosine kNN score prediction: k={5,15,45}, capped at available rows. Choose hyperparameters by tune-only multi-model score MSE. A length-bin score predictor uses six train-only quantile bins; global mean score and all 11 fixed models are controls. Routing maximizes predicted score minus lambda*train-mean normalized model cost for lambda={0,.02,.05,.1,.2,.5,1,2}. No generated response, true quality, actual completion length or realized per-query cost is available to the router. Exact authors' SaveRouter/RouteLLM implementations are not reproduced; no superiority claim against them is permissible.

Tune-only policy selection minimizes upfront/H + mean serving cost, subject to tune excess error <=.02 relative to reference. Each selected policy is then frozen and independently assessed on calibration. Matched policies are used across deployment-decision rules. Study horizons H={10,000,100,000}; router per-request overhead scenarios={0,.01,.1} times train-mean reference-call cost. Judge-cost sensitivity={0,1} times train-mean reference-call cost per observed query/model feedback; correctness scoring itself needs no paid judge in this benchmark. These added charges are assumptions, never measured expenditure.

Upfront cost charges all 11 cached model calls used for training/tune/calibration plus assumed judging cost. This intentionally evaluates a dense-feedback workflow; future sparse-feedback methods must count their own acquisition histories. Include measured router fitting/prediction times separately; no made-up GPU, labor or energy prices. A sunk-data reuse sensitivity multiplies upfront cost by {0,1}; amortization is not invariant to this assumption.

## Decision rules and uncertainty

Point rule: calibration excess error <=.02 and estimated total amortized cost below reference. Bootstrap rule: independently selected policy, paired calibration resampling, one-sided 97.5% upper endpoints for quality loss and cost difference (Bonferroni joint 95% nominal); bootstrap is approximate, not distribution-free finite-sample certification. Finite-sample diagnostic: split exact binomial bounds for signed binary quality, and empirical Bernstein for paired dollar-cost difference under a declared per-call support bound $1. The $1 assumption is checked on accessed observations, not asserted for unseen outputs. Its conservatism is reported; response caps cannot be introduced without reevaluating quality.

Pilot evaluation reports approval rates, quality violations and negative amortized savings among approvals, paired approximate uncertainty for these evaluation estimates, abstention and held-out net savings. Evaluation outcomes are noisy and cannot be called population truth. Pilot workloads share benchmark provenance; seed estimates share evaluation pools and are not independent dataset replications.

## Predefined opportunity and novelty gates

Primary: at H=10,000, zero overhead/judge, fresh acquisition cost, point decisions must show observed false approvals (quality loss >.02 OR non-positive net savings on pilot evaluation) in >=20% of approved cases across at least two workloads, with at least six approvals per workload. The bootstrap assessment must reduce the false-approval fraction by >=50% while retaining >=50% of point approvals. Also require retained approvals in >=2/3 seeds and independent evaluation evidence of favorable quality/savings, rather than simply abstaining everywhere. Report every workload and seed. These opportunity thresholds are a development gate, not a formal significance test.

A separate novelty gate requires a distinct, consequential empirical finding beyond the existing payback paper, confirmed on another independently collected public routing benchmark, or a genuinely new technical result beyond standard mean bounds. No novelty is established by selecting costs/horizons after outcomes, adding standard confidence intervals, or declaring a conservative test better because it refuses to deploy.

Only if BOTH gates pass may fresh data be opened, followed by final evaluation after frozen confirmation. Otherwise diagnose and report the negative result. No complete paper claiming new results will be created without satisfying the scientific claim requirements.

## Closest primary sources

- SaveRouter: https://arxiv.org/abs/2609.37402 ; https://github.com/LAMDA-Model-Reuse/SaveRouter . Already accounts for supervision expenditure, serving savings and break-even volume, and studies sparse model-pool expansion.
- RouterBench: https://arxiv.org/abs/2403.12031 ; https://github.com/withmartian/routerbench ; https://huggingface.co/datasets/withmartian/routerbench . Public cached outcomes and costs.
- LLMRouterBench: https://arxiv.org/abs/2601.07206 . Broad evaluation and strong simple baselines; separate benchmark needed for confirmation if the pilot justifies it.
- Learn then Test: https://arxiv.org/abs/2110.01052 . Independent selection and risk testing are established tools, not claimed new.

## Schema correction before the first completed pilot

The first execution stopped on its initial training-data validation, before fitting or evaluating any router: GSM8K cached scores include .25/.5/.75, not just binary correctness. Preserve these bounded scores exactly; do not round, drop or threshold them. Quality loss now means mean cached-score loss, not necessarily single-response error. Replace the binomial quality diagnostic with the same bounded empirical-Bernstein bound used for a signed [-1,1] variable. Bootstrap and point rules are unchanged. This amendment follows training-only schema inspection, not method-performance inspection. No fresh/final outcome was accessed. The original protocol hash is retained, and a corrected pre-run hash is saved separately.

## Post-pilot engineering sensitivity: economical validation acquisition

The original dense-feedback primary scenario yielded no short-horizon approvals, so its opportunity gate failed. Before drawing deployment conclusions, additionally charge only the reference and the selected policy's distinct call for each independent calibration query; a frozen policy does not need all 11 model outputs on calibration. Keep dense training/tuning, every original primary result and the original gate unchanged. This is a post-pilot cost-accounting sensitivity, not a new primary objective or a new algorithm. Every original outcome is archived before rerunning, and the unchanged dense decisions must reproduce exactly. No learned router or held-out observation is changed. This avoids implying that an unnecessarily expensive calibration workflow is unavoidable.

## Stronger simple-baseline audit

The RouterBench paper explicitly motivates query-independent convex model mixtures as a strong baseline. Add reference/alternative mixtures at fixed fractions {.05,.1,.2,.4,.6,.8}, with an independent prompt-hash draw, for each of the ten other models. This adds 60 policies to the original 43, using the same train/tune feedback and no additional held-out rows. Choose among all 103 using the same original tune criterion. Retain all earlier results as a separate baseline-development attempt. This extension can change the selected policy and is not claimed as exact reproduction of the earlier 43-policy study. It strengthens the control rather than claiming novelty for mixtures. The dense primary gate is evaluated anew without lowering its thresholds.
