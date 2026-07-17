# PRPG

PRPG is a private, auditable plausible-return path generator for three semantic
asset roles: global equity (`ACWI`), municipal bonds (`MUB`), and taxable bonds
(`LQD` in the current version 5; `AGG` in historical version 4). Its governing
design is
[`doc/PRPG-technical-design-and-implementation-log.md`](doc/PRPG-technical-design-and-implementation-log.md).

Recipients of the self-contained version-5 package should start with
[`ONBOARDING.md`](ONBOARDING.md). It covers offline installation, the current
architecture, return schemas, quantitative assumptions and statistics,
limitations, appropriate uses, and supported customization.

The current canonical delivery is one population of 5,000 synchronized
50-year daily paths. Monthly, quarterly, and annual returns for those same
paths are derived exactly from the daily log-total returns; weekly output is
not part of version 5. See the concise
[`version-5 user and maintenance guide`](doc/PRPG-v5-user-and-maintenance-guide.md)
for verification, schemas, path roles, commands, fingerprints, customization,
and known limitations. The historical version-1 through version-4 record is
preserved below and in the primary technical log.

## Status

Delivery version 5 is the current result. Frozen source commit
`07806dfba940be4f19f9d783e989f1c5354ea123` generated
`runs/v5/v5-production-5000-20260716` with nine workers: 5,000 paths, 400 CSVs,
67,150,736 rows, and 5,838,093,246 CSV bytes. All hard production checks passed,
including 400/400 hashes, exact replay and compounding, no complete-path
duplicates, volatility ratios within `0.90–1.10`, maximum correlation error
`0.001420`, and relative covariance error `0.010213`. The separate tails,
drawdowns, ACF, source-reuse, and bounded strategy diagnostics are report-only
and do not silently become release gates. That bounded report is also complete:
it found 24/24 unique sampled paths, zero repeated exact six-month windows in
14,280 sampled windows, all 219 source months represented, and disclosed that
four of six price-only probes looked more favorable in the small simulated
sample than in the single direct historical control.

The paragraphs below preserve the development history and version-4 benchmark;
they do not supersede the current version-5 guide.

Phases 0–2 and the canonical offline data snapshot are complete. The original
version-1 dependence contract is retained as a documented scientific non-pass:
its daily PPW estimate was 108 sessions against a preregistered cap of 60.
Version 2 permits a mean intended daily block length through 126 sessions,
uses scientific stream code 6, and passed its read-only preliminary monthly,
daily, and macro selectors. Its exact-start, nine-worker noncanonical
diagnostic found no admissible K=2..5 design HMM under the then-binding
identification screens, so the reduced rehearsal and resource projection did
not complete. That diagnostic remains immutable and is not a canonical
version-2 G3 result.

The owner has now finalized the exact primary `K=4` HMM after reviewing its
complete 193-month assignment. It produces four intuitive investment regimes:
acute contraction/credit crisis, high inflation/rate stress, steep-curve
recovery/easy-policy expansion, and flat-curve/tight-credit risk-on. The exact
fit is design-main restart 43, seed `2529155491`, with state counts
`[11,26,95,61]`.

The final frozen version-3 software gate is complete. It accounted for all
2,708 collected tests: 2,705 passed in the full command and the three tests
whose data paths depend on the working directory passed from the dedicated
runtime without a source change. Branch-enabled aggregate coverage was 87.66%,
above the approved 80% floor; Ruff and strict mypy passed.

The first full noncanonical rehearsal legitimately did not pass its resource
gate because a normal short-lived-worker exit was classified as incomplete RSS
telemetry; that report remains unchanged. The final rehearsal from frozen
commit `bd11ed8` traversed all 26 registered tasks, verified deterministic
`hmm`, `kernel`, and `bridge` families on nine workers, and passed the resource
projection. It projected canonical G3 at 85,159.56 seconds (23.66 hours),
9,149,177,600 peak bytes (53.26% of 16 GiB), and 12,884,901,888 required free
disk bytes against 84,061,728,768 available.

Canonical version 3 launched once at 2026-07-15 20:24 MDT from frozen commit
`bd11ed8`. It remains a legitimate old-contract non-pass, but its failure has
been correctly attributed: restart 10, seed `75985424`, was macro-bootstrap
stability replicate 53, not the primary design model. The owner prospectively
made those HMM research campaigns nonbinding for delivery version 4.

The practical version-4 path is implemented and the requested local production
bundle is complete. It exactly reproduced and persisted the reviewed K4 fit as
compact artifact
`135fb99e9f25b99b4b42ca2e0c83686d548b9ca61655b794098b1c8b4480f160`
(model `bb43d397f2162493eb05b48417ff23dc94388c9050ce3932985980e09ae1075e`),
and exposes `prepare-k4`, `generate-k4`, and `validate-k4`. A concurrent CSV
directory-creation race found by the first host smoke was fixed narrowly while
retaining the symlink/non-directory safety checks. The final unrestricted-host
gate passed Ruff, strict mypy, and all 2,828 tests; exact branch-enabled
aggregate coverage was 88.4381%, above the approved 80% floor.

Representative one-worker and nine-worker 50-year pilots produced byte-
identical consumer CSVs. Frozen commit `98fbd8f750249bee7ed681292b16ae5ccd872b5a`
then generated `runs/practical-k4-delivery-v4` with nine workers in 94.94
seconds: 5,000 LF and 1,000 HF paths, 190 CSV shards, 19,450,000 rows, and
1,671,856,813 CSV bytes. Full streaming validation passed all hashes, schemas,
period sequences, finite values, 11,550,000 cross-frequency aggregation values,
and complete-path uniqueness in 24.37 seconds. The manifest SHA-256 is
`eaec5d4b80ec717617b2a9e2299dcda497e88a0ae1bcfa7460dd7ff58c4c12f9`.

Whole-production and fixed 500-LF/200-HF observable scans found plausible
means, volatility, tails, bond correlation, drawdowns, and temporal dependence,
with zero complete duplicates and repeated-subsequence rates below their
control limits. Lower equity/bond correlation, attenuated absolute-return ACF,
deeper 50-year bond drawdowns, and the report-only ADR-030 alpha diagnostics
remain explicitly documented. The bundle is complete for private local
research; Yahoo/FRED-derived data and outputs are not authorized for external
redistribution.

The corrected lean G5 path is implemented
through strategy scoring, historical controls, grouped statistics,
qualification, a compact authority boundary, production integration, and a
bounded G7 scan. It uses momentum, trend, reversal, taxable-versus-muni
switching, one linear predictor, and one quadratic predictor; correct
historical-control scoring; three small control banks; 10,000 compact
resamples; and targeted injected-defect tests. Exact k-NN/dictionary machinery,
37 endpoints, and the large meta campaign are retired for this version. The
delivery path retains a bounded predictability/plausibility pilot and final
observable-output validation, without the old 26-task HMM evidence graph.

The historical version-2 Phase-3 software gate and preflight remain historical
evidence only. Version 3's final frozen software gate, rehearsal, preflight,
and input-integrity task passed; its scientific model gate did not.

The governing status, exact fingerprints, scientific gates, and append-only
implementation evidence are maintained in the source-of-truth design linked
above. Do not infer release authority from a successful unit test, preflight,
or rehearsal.

Source data are for private local research use. Raw Yahoo/FRED/Moody's
payloads must not be committed, redistributed, or placed in a consumer bundle.

## Supported environment

The production toolchain is CPython 3.11 on macOS. Create an isolated
environment rather than modifying the system or Anaconda Python:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse dist/prpg-*.whl
```

The final package includes its application wheel and exact dependency
wheelhouse for CPython 3.11/macOS arm64. Developers rebuilding outside that
package may instead install `-e '.[dev]'` with network access. All dependencies
are pinned in `pyproject.toml` and `requirements.lock`; do not silently
substitute versions.

## Core commands

```bash
prpg --help
prpg config validate configs/canonical.yaml
prpg doctor
prpg plan configs/canonical.yaml
python -m prpg.g3_launch --preflight-only --config configs/canonical.yaml --json
python -m pytest --cov=prpg --cov-branch
ruff format --check .
ruff check .
mypy src
```

## Practical K4 starting guide

Run from the repository root with the supported environment activated. The
prepare command is deterministic and returns the approved fingerprint shown
below:

```bash
source .venv/bin/activate
prpg prepare-k4 --config configs/canonical.yaml --json
```

For a bounded 200-LF/60-HF, 50-year pilot:

```bash
prpg generate-k4 135fb99e9f25b99b4b42ca2e0c83686d548b9ca61655b794098b1c8b4480f160 \
  --output runs/practical-k4-science-pilot-v2 \
  --run-name practical-k4-science-pilot-v2 \
  --config configs/canonical.yaml \
  --lf-paths 200 --hf-paths 60 \
  --lf-shard-size 20 --hf-shard-size 10 \
  --workers 9 --json
prpg validate-k4 runs/practical-k4-science-pilot-v2 --json
```

For the requested full 5,000-LF/1,000-HF delivery geometry, use a new output
directory and keep it unchanged until validation finishes:

```bash
prpg generate-k4 135fb99e9f25b99b4b42ca2e0c83686d548b9ca61655b794098b1c8b4480f160 \
  --output runs/practical-k4-delivery-v4 \
  --run-name practical-k4-delivery-v4 \
  --config configs/canonical.yaml \
  --lf-paths 5000 --hf-paths 1000 \
  --lf-shard-size 100 --hf-shard-size 50 \
  --workers 9 --json
prpg validate-k4 runs/practical-k4-delivery-v4 --json
```

`generate-k4` refuses an existing output root. Choose a new run name/directory
for a new run; do not merge or overwrite two runs. The command above is the
exact command used for the completed delivery. To reproduce it, select a new
output root such as `runs/practical-k4-delivery-v4-replay`.

`prpg config validate` is side-effect free. It rejects unknown keys and invalid
scientific combinations, then reports the canonical SHA-256 configuration
fingerprint. `prpg plan` reports exact structural row and shard counts without
acquiring data. `prpg doctor` never prints secret values.

`python -m prpg.g3_launch --preflight-only` is non-launching: it verifies the
approved configuration, immutable calibration input, scoped G3 source
closure/toolchain, scientific constants, and machine resources. The separate
`--canonical-launch` mode is a one-shot scientific action and may be used only
from the frozen worktree after every prerequisite in the source-of-truth log is
recorded as passed. The legacy all-application `prpg calibrate` route cannot
initialize G3.

## Public package boundaries

- `prpg.config`: immutable typed configuration, canonical serialization,
  hashing, schema generation, and redaction.
- `prpg.simulation.rng`: deterministic SeedSequence/PCG64DXSM stream keys.
- `prpg.data.calendar`: pure ordinal `SC252-52-v1` grouping logic.
- `prpg.errors`: stable operational error taxonomy and exit codes.
- `prpg.cli`: command surface and Phase-1-safe diagnostics.

Scientific functions receive explicit typed inputs; they must not read hidden
global state. Provider access is isolated from transformations and generation,
and every stage after acquisition must be reproducible offline.

## Contribution workflow

1. Read the source-of-truth design and the current implementation log.
2. Make one scoped, reviewable change; never edit immutable artifacts in place.
3. Add one focused regression when the change prevents a concrete production
   failure; do not grow tests solely to raise a coverage number.
4. Use scoped tests during development. At a designated software gate, run the
   full suite once with formatting, linting, strict typing, and branch-enabled
   coverage. Version 3 uses one aggregate Coverage.py floor of 80%; the
   statement, branch, and module figures remain useful diagnostics rather than
   additional gates.
5. Record design deviations, commands, evidence, fingerprints, and outcomes in
   the append-only implementation log.

Ordinary geometry customization belongs in a copied
`configs/v5-configured-example.yaml` plus its seed registry and uses the
`configured` profile. Path count, whole-year horizon, role split, shard size,
workers, output root, and noncanonical path seed do not require a model refit.
Ticker/history or supported K4 numerical-setting changes do; state count,
covariance family, schema, frequency, calendar, or asset-count changes require
a new application/scientific version. See `ONBOARDING.md` for the exact matrix.
