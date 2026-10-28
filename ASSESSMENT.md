# Public LLM lifecycle-routing study: executed pilot and novelty assessment

**Status: the predefined opportunity gate failed, and no defensible novel contribution is established.** This is a reproducible public-data engineering study, not a new routing method or a complete research paper.

## The useful engineering question

When does a learned router recover the cost of obtaining its feedback and validating its quality, relative to using an already-deployed reference model? This is relevant to AI engineering, but the core upfront-cost and payback framing already appears in *Routing Should Pay for Itself: Sparse Supervision for Economical LLM Routing* (SaveRouter, September 2026): https://arxiv.org/abs/2609.37402 . We therefore narrowed this pilot to prospective deployment decisions under independent quality/cost uncertainty. Standard bounds and paired bootstrap are not claimed as new statistical contributions.

## Public data and controls

Downloaded the official 99.6 MB zero-shot RouterBench cache (36,497 queries; 11 models) from the pinned Hugging Face revision in source.json. The original benchmark is documented at https://arxiv.org/abs/2403.12031 . Only the preselected math, reasoning and law workloads are analyzed. All output columns, costs and quality come from historical cached runs; no contemporary LLM was called and no API inference expenditure was incurred in this study.

The training-only schema check found fractional scores such as .25/.5/.75 in GSM8K, so we preserve them. Mean cached-score loss is not always a single-response error rate. No values are rounded, thresholded, clipped to force binary outcomes or removed for poor performance. The declared support [0,1] is checked for accessed scores, and [0,$1] for accessed historical per-call costs.

Exact normalized prompt duplicates are grouped before stable hash splitting: 30% training, 10% tuning, 20% calibration, 20% pilot evaluation, 10% fresh and 10% final. The schema-inspected first prompt is assigned to training. Every held-out group stays outside model fitting, vocabulary construction, tuning, calibration and pilot inference. The program audits these separations. Public models may have pretraining contamination; this cached benchmark does not resolve it. Seeds share calibration/evaluation pools, so they are training perturbations rather than independent dataset replications.

Router families are TF-IDF ridge, cosine kNN, prompt-length bins and global mean score, with all 11 fixed models as controls. Every learned family receives the same training feedback, and ridge/kNN each receive three tune-only settings. Eight fixed quality/cost weights initially produce 43 policies. A stronger-baseline audit adds 60 query-independent reference/alternative mixtures at predefined fractions, giving 103 policies per final run. This control is motivated by RouterBench's Zero Router. Its draws use an independent prompt hash and never outcomes. Model choice uses prompt-derived features and training-average costs; generated responses, oracle routing labels and actual completion costs are unavailable at routing time. Candidate selection uses only tuning data. All decision rules assess the identical frozen policy on independent calibration data. GPT-4-1106-preview is the predetermined reference.

These are meaningful simple baselines, but not an exact reproduction of SaveRouter, RouteLLM or modern embedding routers. No experimental superiority over those papers is asserted.

## Executed conditions

18 training runs: three workloads x three seeds x two requested training budgets (256/1024, capped at available groups). We retain all runs and all sensitivity combinations: 10K/100K deployment horizons, 0/1%/10% modeled overhead, 0/1-reference-call modeled judge cost per feedback item, and fresh-versus-reused feedback. Added overhead and judge prices are assumptions, not measurements. Actual model fit-and-prediction seconds are separately recorded in each manifest entry.

The initial predeclared workflow acquires all 11 models on train, tune and calibration. A more economical post-pilot sensitivity charges only the selected policy and reference on calibration, deduplicating a shared reference call. Training and tuning remain dense. All original 43-policy dense decisions reproduced exactly after this sensitivity was added. The subsequent 103-policy stronger-baseline audit can change selected policies and is preserved as a separate attempt; it is not claimed to reproduce the original selections. Neither extension is a new algorithm. In particular, the study does not imply dense validation acquisition is necessary.

For each frozen candidate, the point decision requires mean calibration score loss <=.02 and positive estimated net savings after upfront amortization. The bootstrap rule uses paired one-sided 97.5% upper endpoints for score loss and cost difference, providing only approximate per-decision joint uncertainty. It is not a finite-sample guarantee. The bounded diagnostic uses known empirical-Bernstein machinery; its $1 support assumption is not a proven guarantee for future uncapped service outputs. The finite-bound rule may be highly conservative, and zero approvals cannot demonstrate usefulness.

## Main results and engineering sensitivity

Fresh feedback, zero modeled overhead and zero modeled judge fee:

| Validation acquisition | Horizon | Workload | Point approvals / 6 | Bootstrap approvals / 6 | Observed bad point approvals |
|---|---:|---|---:|---:|---:|
| dense | 10,000 | math | 0 | 0 | 0 |
| dense | 10,000 | reasoning | 0 | 0 | 0 |
| dense | 10,000 | law | 0 | 0 | 0 |
| dense | 100,000 | math | 1 | 0 | 0 |
| dense | 100,000 | reasoning | 0 | 0 | 0 |
| dense | 100,000 | law | 6 | 0 | 0 |
| selected_pair | 10,000 | math | 1 | 0 | 0 |
| selected_pair | 10,000 | reasoning | 0 | 0 | 0 |
| selected_pair | 10,000 | law | 0 | 0 | 0 |
| selected_pair | 100,000 | math | 1 | 0 | 0 |
| selected_pair | 100,000 | reasoning | 0 | 0 | 0 |
| selected_pair | 100,000 | law | 6 | 0 | 0 |

“Observed bad” means a pilot-evaluation point estimate breaches either the quality-loss target or positive amortized savings. It is NOT a population truth label. The raw outputs also retain stricter “clearly bad/good” indicators using independent approximate pilot intervals. Six decisions per workload share observations and should not be treated as six independent production deployments.

Under the predeclared 10,000-query dense-feedback scenario, the point rule already refuses every deployment. Therefore this pilot does not demonstrate the proposed widespread false-approval phenomenon, and cannot establish a benefit from adding uncertainty checks. Across the broader predefined cost sensitivities, some adverse point approvals occur, but the bootstrap and finite-bound rules reject every deployment. This is not enough to claim a useful, novel controller: retaining useful approvals was part of the gate. Results from the selected-pair sensitivity and sunk-data scenarios are all in deployment_decisions.csv; none substitutes for the failed primary gate.

The stronger baseline matters: in the archived 43-policy study, the law workload at 100K queries (fresh costs, zero overhead/judge) had five point approvals, three with adverse pilot point estimates. Adding the fixed-mixture controls changes the selected law policies: all six point approvals then have favorable pilot point estimates in that same configuration. Their quality uncertainty is still substantial, so this is not a safety claim. It shows why the earlier apparent failure mode cannot be presented as a robust research contribution without strong simple controls.

Measured runtime for the complete last sweep was 33.55 seconds excluding imports. It produced 2,592 scenario/rule decisions from 18 fitted runs. Four smoke tests passed before the first completed run, and the final five-test suite additionally checks stable input-only fixed mixtures. The report audits exact reproduction of original dense decisions at the earlier sensitivity stage, split isolation, model-index counts and that economical validation never increases acquisition cost.

## What is and is not novel

The public-data workflow is useful and reproducible, but payback accounting is established by SaveRouter; independent policy assessment and confidence bounds are established inference; and a more economical calibration feedback plan is straightforward once the policy is frozen. These pieces do not become novel merely because they are combined.

An empirical contribution would require a consequential, previously unestablished deployment failure mode that survives strong baselines, multiple independent benchmarks, uncertainty analysis and prospective confirmation. This pilot has not established that. It also does not establish that lifecycle-aware routing research is impossible or that no router is worthwhile. The evidence is conditional on historical models, three workloads, dense train/tune acquisition, the selected budgets, score proxy and assumed horizons.

## Limits and attempted alternatives

The original binary-score assertion failed before model fitting. It was corrected using training-only schema information; the amendment and both protocol hashes are retained. After the primary gate failed, the selected-pair validation sensitivity removed unnecessary calibration calls while preserving every original result. The final run strengthens the query-independent baseline through fixed mixtures, preserving earlier runs in a separate folder. These are documented exploratory alternatives, not favorable-result selection.

No real serving latency, GPU memory, throughput, energy, labor, contemporary prices, retry effects or model-maintenance traces are measured. Cache replay supports quality and historical call-cost comparisons only. It does not validate a complete production lifecycle or guarantee future quality. The $1 support bound and modeled judge/overhead fees are explicitly conditional. Known TF-IDF and benchmark-provenance limitations prevent broad deployment conclusions.

The fresh and final pools remain unused. The full-run guard was exercised separately and refuses evaluation until there is a passing pilot plus a defensible scientific claim. No complete paper claiming novelty was produced.

## Reproduction

See README.md for exact clean-environment commands. The source downloader pins the revision and the reader checks SHA256 before loading. The deliverable excludes the raw dataset because its metadata does not specify a redistribution license. Download from the authors' public source to reproduce.
