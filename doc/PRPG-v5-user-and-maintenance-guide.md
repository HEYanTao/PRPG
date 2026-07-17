# PRPG delivery version 5 — user and maintenance guide

## 1. Delivered application and data

Delivery version 5 is the current PRPG result. It generates one deterministic
population of 5,000 synchronized 50-year return paths for three investment
roles:

| Consumer column | ETF proxy | Meaning |
|---|---|---|
| `equity_log_return` | ACWI | Global equity total return |
| `muni_bond_log_return` | MUB | U.S. investment-grade municipal-bond total return |
| `taxable_bond_log_return` | LQD | U.S. investment-grade corporate-bond total return |

Daily log returns are authoritative. Monthly returns are exact sums of daily
logs, quarterly returns are exact sums of three monthly logs, and annual
returns are exact sums of twelve monthly logs. The K4 HMM controls the sequence
of investment environments; each simulated month then copies one complete,
synchronized historical ACWI/LQD/MUB calendar-month block from that state.

The completed canonical run is:

```text
runs/v5/v5-production-5000-20260716
```

It contains 5,000 paths, 100 work units, 400 consumer CSV files, 67,150,736
rows, and 5,838,093,246 CSV bytes (about 5.44 GiB). The production hard
validator passed all frozen structural and scientific gates.

## 2. Quick verification

Run these commands from the repository root. They do not modify the CSVs.

```bash
cd runs/v5/v5-production-5000-20260716
shasum -a 256 --status -c checksums.sha256
cd ../../..
.venv/bin/prpg v5 validate \
  runs/v5/v5-production-5000-20260716 \
  --profile canonical \
  --config configs/v5-canonical.yaml \
  --json
```

The checksum command should exit silently with status zero. The validator
should report `"status":"passed"`, `5,000` paths, `400` verified CSVs, and
`67,150,736` rows. On the production Mac, full validation took about 208
seconds.

The validator streams every consumer file and independently checks:

- the manifest, file allowlist, SHA-256 list, schemas, path IDs, period order,
  row counts, and finite values;
- the independently reconstructed hidden-state and source-month draws;
- exact daily-to-monthly-to-quarterly-to-annual log compounding;
- absence of duplicate complete daily paths;
- ensemble volatility, correlation, covariance, state occupancy, and state
  transition limits; and
- absence of source-month, state, or future-return columns in consumer CSVs.

## 3. Output layout and path roles

```text
v5-production-5000-20260716/
├── manifest.json
├── checksums.sha256
├── validation-report.json
├── diagnostics-report.json
├── software-coverage.json
└── consumer/returns/
    ├── training/
    │   ├── daily/
    │   ├── monthly/
    │   ├── quarterly/
    │   └── annual/
    ├── validation/
    │   ├── daily/
    │   ├── monthly/
    │   ├── quarterly/
    │   └── annual/
    └── test/
        ├── daily/
        ├── monthly/
        ├── quarterly/
        └── annual/
```

Each shard contains 50 complete paths. The partition is fixed and visible in
both directory names and path IDs:

| Role | Path IDs | Paths | Shards per frequency |
|---|---:|---:|---:|
| Training | `P000001`–`P004000` | 4,000 | 80 |
| Validation | `P004001`–`P004500` | 500 | 10 |
| Test | `P004501`–`P005000` | 500 | 10 |

Do not mix validation or test paths into RL training. A quarterly agent may use
only information available through the completed prior quarter. Version 5 does
not certify within-month daily trading because a daily agent could recognize a
reused historical source block.

## 4. CSV schemas and return interpretation

Daily CSV header:

```text
path_id,period_index,simulation_year,model_month,session_in_month,equity_log_return,muni_bond_log_return,taxable_bond_log_return
```

Monthly, quarterly, and annual CSV header:

```text
path_id,period_index,simulation_year,period_in_year,equity_log_return,muni_bond_log_return,taxable_bond_log_return
```

Returns are decimal total log returns, not percentages and not simple returns.
For example, `0.01` is a log return of 0.01, whose corresponding simple return
is `exp(0.01) - 1`, approximately 1.005%. For any consecutive period, compound
by summing log returns:

```python
simple_cumulative_return = math.exp(sum(log_returns)) - 1.0
```

Every path has exactly 600 model months, 200 quarters, and 50 years. Daily row
counts vary because a sampled calendar month retains its actual 19–23 common
trading sessions. In production the daily count ranged from 12,467 to 12,704
per path, with a mean of 12,580.1472. Consumers must use `model_month` and
`session_in_month`, not assume 252 sessions per simulated year.

The exact production totals are:

| Frequency | Rows | CSV files | Bytes |
|---|---:|---:|---:|
| Daily | 62,900,736 | 100 | 5,492,694,638 |
| Monthly | 3,000,000 | 100 | 245,327,685 |
| Quarterly | 1,000,000 | 100 | 80,448,571 |
| Annual | 250,000 | 100 | 19,622,352 |

There is intentionally no weekly version-5 output.

## 5. Exact production identity

Preserve these identities together. A different value means a different input,
model, RNG namespace, simulation, materialization, toolchain, or run.

| Identity | SHA-256 or commit |
|---|---|
| Frozen generator source commit | `07806dfba940be4f19f9d783e989f1c5354ea123` |
| Dependency lock | `bd03d5d02faf50e5acff83f5c1489d93a701714af7387db153eec8f5fe783d7c` |
| Configuration | `b903100a9e4c322b1ac20555272de32c4666d9625ef408ce459c5c8ec8081466` |
| Seed registry | `2c58077db1196b97866d39c5445d093047cc84b0402e1e45ed6d0e596c34956c` |
| Processed month library | `e2caa7756ec5f6b37806e6ef58e44fe827f741ed544e82b132a0cd2f1bbe1760` |
| Approved K4 model | `be5df93d382a499dbf21c00b1e0be9d47bfda08bd387d0ac70b36d47ba88d6c7` |
| Simulation | `9f432a86045f7fd0e0db02310fb4214775489cf5d30703704d677b3018c5fead` |
| Materialization | `76b5039fabe7fc1e98832e3c9d67e21ec1c66753dfaf0281b7e04d7912b0e7c6` |
| Run | `101600b117d07921aee8589f25da8dcff0f2ab80c0c6db7e692040a929283705` |
| Internal manifest identity | `d35a50a59035066d124cffe0829fc2a1405c06ef480bf10e02395752c6b7541f` |

File hashes for the delivery metadata are:

| File | SHA-256 |
|---|---|
| `manifest.json` | `ce3edc2494151d1007e33919cc61ea6acc5afc29887c61965b45bfcced23339c` |
| `checksums.sha256` | `b9fd3951b81914772e1f0478318620f3fbf3092ee106897a3912e2218a2fb202` |
| `validation-report.json` | `53d7b491b8794f51d35017f386b841350c0a44b6381ebd0fea52346a1e58ef20` |
| `diagnostics-report.json` | `7eb46a5e667daee3e03643dd5f4e2b36349bc3b47d31240594f8c379c93d81cb` |
| `software-coverage.json` | `d607d72c167783e8964b6e67394d6f5c305b1581122b687f14965a3b2f423407` |

The manifest file hash and the internal manifest identity are intentionally
different: the internal identity covers the manifest's identity payload before
its self-describing publication fields are added.

## 6. Reproducing or creating a new run

Use CPython 3.11 on macOS. The existing `.venv` is the exact environment used
for production. To rebuild an environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse dist/prpg-*.whl
```

Canonical generation requires a clean Git worktree, the verified local data
and model artifacts, at most nine workers, and a fresh output directory that
does not already exist. The exact production command was:

```bash
.venv/bin/prpg v5 generate \
  --model be5df93d382a499dbf21c00b1e0be9d47bfda08bd387d0ac70b36d47ba88d6c7 \
  --processed e2caa7756ec5f6b37806e6ef58e44fe827f741ed544e82b132a0cd2f1bbe1760 \
  --output runs/v5/v5-production-5000-20260716 \
  --run-name v5-production-5000-20260716 \
  --workers 9 \
  --profile canonical \
  --config configs/v5-canonical.yaml \
  --json
```

Never point a reproduction at the existing completed root. Choose a new run
name and output root. The generator creates a fresh-only directory and aborts
instead of overwriting an earlier result.

On the 10-core Apple M4 Mac mini with 16 GiB RAM, the canonical generator took
42.89 application seconds (44.34 seconds wall time), 317.67 user CPU seconds,
and 12.34 system CPU seconds with nine workers. Maximum measured resident memory
was 274,956,288 bytes (about 262 MiB), with zero swaps. Full validation took
206.27 application seconds (207.96 seconds wall time). The preflight required
18,752,000,000 free bytes and observed 79,319,740,416; the delivered CSVs use
5,838,093,246 bytes. Retain at least 20 GB free before another canonical run.

For bounded development checks, use the same command with a fresh root and one
of these registered profiles:

| Profile | Purpose | Paths |
|---|---|---:|
| `smoke` | One end-to-end 50-year path | 1 |
| `parity` | Representative one-worker versus nine-worker byte comparison | 90 |
| `pilot` | Full plausibility and resource projection before production | 500 |
| `configured` | Geometry and horizon declared in the selected v5 config | Configured |
| `canonical` | Final delivery geometry | 5,000 |

Validate each root with the matching `--profile`. Do not promote a smoke or
pilot into a canonical result.

## 7. Production validation result

The canonical hard validation passed in 206.2675 seconds:

| Metric | Observed | Frozen requirement |
|---|---:|---:|
| Consumer checksums | 400/400 | All |
| Source replay error | 0.0 | `<=1e-12` |
| Log-compounding error | 0.0 | `<=1e-12` |
| Complete-path duplicates | 0 | 0 |
| ACWI volatility ratio | 1.005043 | 0.90–1.10 |
| MUB volatility ratio | 1.008400 | 0.90–1.10 |
| LQD volatility ratio | 1.005868 | 0.90–1.10 |
| Maximum pairwise correlation error | 0.001420 | `<0.05` |
| Relative covariance error | 0.010213 | `<=0.15` |

Aggregate state occupancy was `[0.138052, 0.204772, 0.429554, 0.227623]`.
Every occupancy and transition estimate passed its frozen sampling-error band.

The final software gate also passed: 2,862 tests in 1,368.25 seconds; Ruff
format/lint and Git diff checks were clean; strict mypy passed 125 source files;
and aggregate branch-enabled coverage was 87.6980%, above the approved 80%
floor.

### Report-only diagnostics

The final bounded diagnostic report is complete and remains explicitly
nonblocking: `status="diagnostic_report_only"`, `release_gate=false`, and
`decision=null`. Reproduce it with:

```bash
.venv/bin/prpg v5 diagnose \
  runs/v5/v5-production-5000-20260716 \
  --config configs/v5-canonical.yaml \
  --json
```

The declared convenience sample uses 12 development paths and 24 assessment
paths spanning training, validation, and test. It compares 83 monthly and 83
daily tail, drawdown, ACF, covariance, correlation, and spread endpoints with
the direct 2008-04 through 2026-06 historical sequence. All 24 assessed paths
were unique, all 14,280 exact six-month return windows were unique, and all 219
historical source months appeared. Sample state occupancy was
`[0.1272, 0.2022, 0.4420, 0.2286]`.

The six past-return-only strategies use weights chosen after a completed
quarter and held through the following three months. Median simulated alpha
versus the identically calculated, non-zero-filled historical control was:

| Strategy | Simulated median | Historical control |
|---|---:|---:|
| Six-month momentum | 0.1205 | -0.2282 |
| Twelve-month trend | 0.1793 | -0.5024 |
| One-month reversal | -0.0232 | 0.0906 |
| Taxable-versus-muni switch | 0.0391 | 0.2122 |
| Linear ridge | 0.1229 | -0.1999 |
| Quadratic ridge | 0.3846 | -0.0844 |

Momentum, trend, linear, and quadratic probes therefore look more favorable in
this small simulated sample than in the one available historical control,
while reversal and the bond switch look less favorable. This is a disclosed
RL-usefulness limitation and a reason to retain the 4,000/500/500 discipline;
it is not a calibrated hypothesis test, proof of exploitable alpha, or a
post-result release veto. The sample covers only 24 of 5,000 paths, history is
shorter than a simulated path, and six probes are not exhaustive.

## 8. Safe customization workflow

Never edit the completed run, its manifest, or its CSVs. Preserve version 5 as
the reproducible baseline and use a new config, identity, run name, and output
root for future work.

| Requested change | Required workflow |
|---|---|
| Worker count (1–9), run name, or output root | No model refit. Use a fresh root. Worker count may change run metadata but fixed inputs produce the same consumer CSV bytes. |
| New random paths with the same model | Use a new noncanonical root and `prpg v5 generate --path-seed N`. Keep the config/registry master seed unchanged because it identifies the fitted model. The run manifest records the new path seed. |
| Path count, positive role split, whole-year horizon, or shard size | Copy `configs/v5-configured-example*.yaml`, update both geometry and seed-registry role ranges, and use `--profile configured`. No refit is needed while data and model settings remain unchanged. |
| Data cutoff or quality/materialization rule | Acquire and prepare a new immutable month library, review exclusions, refit K4, review all assignments, then smoke/pilot/produce under new fingerprints. |
| Supported K4 numerical setting or one of the three asset proxies | Update a new config, acquire/prepare data when the proxy/history changes, refit K4, and review all assignments before generation. |
| State count, covariance family, asset count, or feature definition | Create a new delivery/scientific version. These remain fixed K=4/diagonal/three-return architecture choices, not ordinary v5 configuration. |
| New frequency such as weekly | Add a new schema/materialization version and exact derivation tests. Weekly is not part of v5. |
| Validation threshold or new hard gate | Define it prospectively before observing the candidate run, tie it to a concrete production failure, and obtain owner approval. Diagnostics remain report-only by default. |

For any meaningful complexity addition, apply the six project guardrails in
Section 30.17 of the primary technical log: identify the consumer-visible
output, critical-path need, concrete prevented failure, production relevance,
diagnostic/gate status, and whether support code is outgrowing scientific logic.

## 9. Maintenance map

The small version-5 implementation surface is:

| Area | Main files |
|---|---|
| Frozen configuration and RNG namespace | `configs/v5-canonical.yaml`, `configs/v5-seed-registry.yaml`, `src/prpg/v5/config.py` |
| Complete synchronized month library | `src/prpg/v5/data.py` |
| Fixed K4 fit and review packet | `src/prpg/v5/hmm.py` |
| Calendar-month simulation and CSV writer | `src/prpg/v5/generation.py` |
| Streaming hard validation | `src/prpg/v5/validation.py` |
| Report-first diagnostics | `src/prpg/v5/diagnostics.py` |
| Operator commands | `src/prpg/cli.py` |
| Focused regression tests | `tests/unit/test_v5_*.py`, `tests/unit/test_cli.py` |

After a scoped change, run the smallest relevant v5 test file plus Ruff and
mypy. Run the full suite/coverage gate once only when preparing a new frozen
production source. Do not create tests merely to push coverage above 80%.

## 10. Troubleshooting and preservation

- **Dirty-worktree error:** commit or deliberately remove unrelated changes;
  canonical generation records and requires a clean source tree.
- **Output root exists:** choose a new root. Do not overwrite or merge files
  into a completed or partial run.
- **Insufficient disk:** free space until the generator's preflight passes;
  retain at least 20 GB free for the current geometry.
- **Worker failure:** preserve the incomplete directory for diagnosis, correct
  the concrete cause, and restart in a new root. Version 5 deliberately has no
  elaborate crash-resume framework.
- **Checksum mismatch:** treat the affected bundle as corrupted. Restore a
  verified backup or regenerate from the frozen inputs; never hand-edit the
  checksum list to make it pass.
- **Plausibility non-pass:** preserve the report and investigate the real
  production cause. Do not tune frozen thresholds after seeing the result.
- **Incomplete data or a K4 pool below 10 months:** stop before generation,
  inspect the data/assignments, and obtain an explicit prospective decision;
  do not impute months or silently substitute K3.

Back up the entire canonical run root, `artifacts/v5`, `configs/v5-*`, the
source repository including commit `07806df`, and this guide. Raw provider data,
derived artifacts, and generated paths are for private local research and are
not licensed here for external redistribution.

## 11. Known limitations

- The empirical library contains 219 complete synchronized months from April
  2008 through June 2026. Five thousand paths do not create new historical
  regimes; source-month reuse is expected.
- K4 states are descriptive return environments, not causal macroeconomic
  claims. The diagonal HMM simplifies classification, while synchronized
  empirical blocks preserve observed within-month joint asset behavior.
- Synthetic boundaries are created between independently sampled months, so
  exact cross-month dependence is not guaranteed beyond the HMM state chain.
- The design is conditional on one fitted K4 model and does not propagate
  parameter or model-selection uncertainty.
- Fifty-year paths naturally admit more extreme drawdowns than the available
  18-year historical window; comparisons must use compatible horizons.
- Daily path lengths vary, weekly output is absent, and only completed-quarter
  decision timing is certified for RL use.
- LQD represents investment-grade corporate bonds, not the broad aggregate
  bond market represented by AGG in delivery version 4.

The full design rationale, historical non-passes, implementation log, and
traceability record remain in
[`PRPG-technical-design-and-implementation-log.md`](PRPG-technical-design-and-implementation-log.md).
