> Historical workspace instructions. Use README.md for this standalone repository.

# Public-data lifecycle LLM routing

This code investigates whether quality/cost uncertainty changes a router deployment decision after charging for feedback acquisition. It uses the official public RouterBench cache, not synthetic LLM outcomes. **No new method or publishable novelty is claimed.** The broad supervision/payback idea is already studied by [SaveRouter](https://arxiv.org/abs/2609.37402).

Read ASSESSMENT.md for executed findings, qualifications and the gate decision. PROTOCOL.md records the original plan and explicitly labeled amendments. Earlier outcome files are preserved rather than replaced by favorable attempts. The raw dataset is downloaded to work/ and excluded from this package.

## Exact commands

Run from the workspace root, the directory containing outputs/ and work/. Python 3.12 was used. Windows PowerShell:

```powershell
python -m venv work/lifecycle-env
.\work\lifecycle-env\Scripts\python.exe -m pip install -r outputs/lifecycle-routing/requirements-lock.txt
.\work\lifecycle-env\Scripts\python.exe outputs/lifecycle-routing/fetch_data.py
.\work\lifecycle-env\Scripts\python.exe -m unittest discover -s outputs/lifecycle-routing -p test_study.py -v
.\work\lifecycle-env\Scripts\python.exe outputs/lifecycle-routing/study.py --stage pilot
.\work\lifecycle-env\Scripts\python.exe outputs/lifecycle-routing/report.py
```

Linux/macOS: replace `.\work\lifecycle-env\Scripts\python.exe` with `work/lifecycle-env/bin/python`. The default dataset location is work/lifecycle-routing/routerbench_0shot.pkl; override with `--data PATH` if necessary. The loader checks a pinned SHA256 and uses an explicit constructor allowlist for the official pandas serialization.

The completed pilot model sweep takes approximately one minute on this CPU, excluding downloads/imports and first-time font caching. The public download is about 100 MB. No GPU, paid API calls, model credentials, model-weight downloads or private data are required. The reported replay expenditures are hypothetical historical costs of acquiring the cached feedback, not dollars actually spent in this session.

## Outputs

- results/pilot/raw_decisions.json: every horizon, cost assumption, rule, seed and budget.
- selected_policies.json: all selected families, quality differences, costs and uncertainty.
- manifest.json: model list, exact split indices, chosen model indices on calibration/pilot, tune scores, model timings and source/protocol hashes.
- gate.json: predefined primary opportunity decision; novelty is assessed separately.
- results/report/deployment_decisions.csv and policy_metrics.csv: exportable tables.
- results/report/break_even.pdf and break_even.png: standalone scientific figures.
- results/report/verification.json: isolation and reproduction audit.
- source.json: official data revision, URL and checksum.

The saved earlier 43-policy pilot and cost-accounting sensitivity can be inspected in results/pilot/before-*/. Current code includes the stronger 103-policy candidate pool. Rerunning it does not regenerate the archived earlier implementation; those outcomes are retained provenance, not claimed current-code outputs.

## Scientific limits

Public cached outcomes establish historical quality/call-cost comparisons. They do not measure present-day serving latency, energy, throughput, engineering labor, current prices or a production maintenance lifecycle. Added judge/overhead fees are labeled assumptions. Score loss may be fractional; no score is converted into a binary label to make a preferred statistical test work.

TF-IDF and simple routing baselines are not exact reproductions of state-of-the-art author code. The paired bootstrap is approximate, and the finite-bound diagnostic relies on declared bounded-cost/IID assumptions. Seeds share calibration/evaluation pools. Refusing every deployment is not evidence that an uncertainty method is useful.

Fresh/final evaluation is deliberately closed:

```powershell
.\work\lifecycle-env\Scripts\python.exe outputs/lifecycle-routing/study.py --stage fresh
.\work\lifecycle-env\Scripts\python.exe outputs/lifecycle-routing/study.py --stage final
```

These commands correctly refuse while the pilot and scientific-claim requirements are unmet. Do not lower the gate or relabel used pilot data as fresh confirmation. A defensible new claim requires independent benchmark evidence or new theory, not simply more sweeps of this cache.
