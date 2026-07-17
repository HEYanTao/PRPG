# Changelog

All notable implementation changes are recorded here. Scientific decisions,
failed experiments, gate evidence, and token checkpoints additionally belong
in the append-only source-of-truth implementation log.

## [Unreleased]

### Added

- A self-contained version-5 delivery workflow with an offline macOS-arm64
  wheelhouse, application wheel, explicit package allowlist, package-wide
  checksums, isolation verifier, primary quant-developer onboarding guide, and
  full-population historical-versus-simulated comparison.
- A `configured` v5 profile and runnable 12-path example for changing positive
  role splits, whole-year horizons, shard sizes, and worker counts without
  altering the canonical 5,000-path contract.
- A noncanonical `--path-seed` override that creates a new deterministic path
  population while leaving the fitted model's calibration seed unchanged; the
  new seed is bound into generation identities, manifests, and validator replay.
- Phase-1 installable `src/prpg` package and full planned CLI command surface.
- Strict immutable configuration models, safe YAML loading, canonical JSON
  hashing, redaction, and canonical/smoke configurations.
- Version-1 SeedSequence/PCG64DXSM random-stream key contract and registries.
- Pure `SC252-52-v1` ordinal calendar implementation.
- Stable error/exit-code taxonomy and focused contract tests.
- Exact dependency pins for the supported CPython 3.11/macOS environment.
- Canonical G2 raw/processed data pipeline and immutable offline calibration
  input for 217 monthly and 4,547 daily synchronized return vectors.
- Closed, generation-disabled G3 calibration orchestration and nine-worker
  deterministic scientific primitives.
- A registered, explicitly noncanonical version-3 G3 rehearsal launcher that
  writes an immutable reduced-geometry report and can never mint canonical or
  generation authority.
- Typed, immutable per-K HMM inadmissibility evidence that lets the registered
  K=2..5 family finish evaluating scientific rejections while unexpected
  operational and integrity failures still abort.
- Owner-approved prospective version-3 fixed-`K=4` HMM contract, including an
  exact rare-regime support rule, historical-result preservation, and explicit
  launch-authority guardrails. The bounded core implementation now follows this
  contract, but no accepted model or canonical result exists.
- A dedicated `python -m prpg.g3_launch` composition root and versioned G3
  execution-closure manifest so calibration provenance excludes unrelated
  downstream validation/release modules while retaining exact source, static
  resource, seed, runtime, data, and dependency identities.
- Lean G5 price-only strategy scoring for momentum, trend, reversal,
  taxable-versus-muni switching, linear ridge, and quadratic ridge, with the
  same exposure-controlled alpha calculation for simulated and historical
  paths.
- Lean G5 inverse-volatility timing, compact three-bank historical controls,
  simultaneous bootstrap statistics, return/state diagnostics, direct
  duplicate/subsequence scans, and a separate qualification-domain permit.
- A verified G3 artifact handoff that copies one completed isolated calibration
  into a clean downstream runtime and reuses the existing scientific loader.
- A compact practical-K4 artifact store and exact reproduction path for the
  owner-reviewed design-main restart 43 model. The persisted artifact is
  `135fb99e...0f160` and its numerical model fingerprint is
  `bb43d397...1075e`.
- Public `prpg prepare-k4`, `prpg generate-k4`, and `prpg validate-k4`
  commands, including deterministic nine-process production coordination,
  concise manifests/checksums, and bounded streaming structural validation.
- Focused practical-delivery tests covering the artifact, CLI, serial binding,
  multiprocessing runner, CSV publication, structural validator, and corrected
  lean strategy scorer. The combined focused command passes 168 tests.
- The complete private delivery bundle at `runs/practical-k4-delivery-v4`:
  5,000 LF and 1,000 HF fifty-year paths, 190 CSV shards, 19,450,000 rows,
  full checksums/receipts, and a compact durable production-science report.

### Changed

- Kept the delivered HMM architecture fixed at K=4 with diagonal covariance
  while exposing its numerical restart count, priors, floors, convergence,
  standardization, and minimum-pool settings for new reviewed fits. Geometry-
  only config changes can reuse a model when its scoped fit provenance matches.
- Made the six owner-directed anti-overengineering questions binding project
  setup: diagnostics remain report-only by default, and added complexity must
  protect a concrete consumer-visible result on the real critical path.
- Activated prospective scientific version 2 while permanently retaining the
  version-1 `L_raw=108 > 60` result as a non-pass.
- Raised only the daily *mean intended* stationary-bootstrap block bound from
  60 to 126 sessions; the monthly bound remains 12 and values above either cap
  still fail without truncation or waiver.
- Allocated fresh version-2 random-stream codes 6–10 and rotated G3,
  block-policy, configuration, and calibration-input identities.
- Set the version-2 software-coverage floors to 80% independently for
  repository-wide statements, repository-wide branches, the combined
  branch-enabled result, and designated critical-module lines. Version 3 later
  simplifies this to one 80% aggregate branch-enabled Coverage.py gate, with
  the component figures retained as diagnostics.
- For version 2, re-versioned the noncanonical rehearsal/resource contract to
  bind the actual benchmark K, forbid rejected candidate parents, and
  conservatively scale lower-K Tier-A timing to K=5-equivalent work.
- Prospectively fixed the version-3 HMM at `K=4`. K=2..5/BIC selection and the
  former posterior-mass, Viterbi occupancy/count, and historically observed
  transition-entry/exit graph floors are report-only for version 3. The exact
  replacement support gate requires at least two observed monthly episodes
  and two daily runs per state while retaining the existing eligible-start,
  effective-length, stability, holdout, adequacy, bridge, G5, and G7 gates.
- Registered version-3 scientific stream codes 11–15, activated scientific
  version 11 for the historical-vector policy, added explicit fixed-K4
  configuration fields, and rotated the G3/block/rehearsal/resource contracts.
  The fixed-K4 success path, recurring-history checks, production refit,
  bridge, and canonical-boundary fences are now integrated.
- Reinforced the Section 4.7 scope freeze: no new G3 gates, no expansion of the
  existing fixed-K4 bridge, rare/unexpected cases abort cleanly, and G4–G8
  quality work is centered on observable simulator outputs.
- Corrected the rehearsal host sampler's treatment of a normal process-lifecycle
  race: a child that exits after enumeration contributes zero current RSS and
  no longer makes RSS telemetry incomplete. Access-denied and operating-system
  failures remain fail-closed.
- Superseded downstream-first G3 sequencing. The exact G3 closure now freezes in
  an isolated commit/worktree and can run while G5 evolves separately; the
  frozen source tree is never modified during execution.
- Replaced the unapproved 37-strategy/large-reservoir G5 proposal with six
  intuitive price-only attack families, correct historical-control scoring,
  one grouped alpha decision, three bounded control banks, 10,000 compact
  resamples, and targeted injected-defect integration. k-NN, dictionary,
  37-endpoint, K3-meta, and 1,000-by-12 campaigns are retired for this version.
- Froze the final G3 execution closure at commit `bd11ed8`, passed its single
  2,708-test/87.66%-aggregate software gate, final 26-task nine-worker
  rehearsal, and nonlaunching preflight, then launched canonical run
  `9eb50787...44203f` once from the isolated runtime.
- Closed canonical version 3 as a legitimate fixed-K4 scientific non-pass
  after the immutable run stopped at the retained posterior-identification
  gate. No retry or post-result threshold waiver was applied; only a
  prospective regime redesign can resume G3.
- Corrected the version-3 failure attribution: restart 10 / seed `75985424`
  was design macro-stability bootstrap replicate 53, not the primary K4 fit.
  The primary design fit is restart 43 / seed `2529155491`, counts
  `[11,26,95,61]`, and exactly reproduces the four month groups reviewed by
  the owner.
- Prospectively finalized that exact primary K4 labeling for delivery version
  4. Posterior-identification, rolling/macro HMM stability, HMM holdout and
  adequacy campaigns, a production HMM refit, and the fixed-K bridge are now
  report-only or deferred. Macro-only/no-leak inputs, deterministic numerical
  usability, conditioned synchronized return sampling, observable
  predictability/plausibility checks, compounding, schemas, counts, and hashes
  remain required.
- Fixed the narrow concurrent-directory creation race exposed when separate
  CSV workers first created a shared consumer/frequency directory. A losing
  worker now accepts `FileExistsError`, then applies the existing safety check
  so a symlink or non-directory winner still fails closed.
- Defined zero timing alpha only for a strategy proven to have exactly static
  weights plus a zero intercept and numerically zero regression residual.
  Dynamic or otherwise degenerate regressions still abort. This occurred for
  68/100 low-frequency and 21/40 high-frequency linear-control fits in the
  pilot and is an estimand correction, not zero-filling of arbitrary scores.
- For delivery version 4, changed the universal absolute `+/-0.10` alpha-
  neutrality cutoff to a report-only diagnostic. The same cutoff rejects
  actual historical scores, while permutation probes show that the strongest
  pilot signals respond to the owner-approved K4 temporal ordering. Historical-
  relative scores and candidate-versus-control gaps remain explicitly reported.
- Froze the final practical application at commit `98fbd8f`, passed the single
  2,828-test/88.4381%-aggregate host gate, and completed nine-worker production
  in 94.94 seconds. All-output validation passed in 24.37 seconds with zero
  complete duplicates; the fixed 500-LF/200-HF observable/repetition scan also
  passed with the ADR-030 alpha warnings retained transparently.

### Scientific diagnostics

- Canonical version 3 exited with model code 20 inside design macro-stability
  bootstrap replicate 53.
  Mean maximum posterior was `0.6384672589805506 < 0.65`, two assigned-state
  posteriors were below `0.60`, and minimum pairwise Mahalanobis separation was
  `0.15852128288523074 < 0.75`. Effective state masses were approximately
  `[1, 2, 92, 98]` months in that synthetic refit. Those values do not describe
  the primary design K4, whose counts are `[11,26,95,61]`. The run produced no
  published model artifact, block result,
  G3 evidence, generation authority, or returns.

- The authorized version-2 preliminary selector passed at monthly/daily/macro
  lengths 10/108/20 against bounds 12/126/24.
- A later exact-start, nine-worker noncanonical diagnostic rejected every
  K=2..5 design HMM under unchanged identification screens. It created no
  canonical result, rehearsal report, resource projection, model artifact, or
  generation authority; canonical G3 remains unlaunched.
- The owner subsequently selected fixed `K=4` prospectively for version 3.
  This decision preserves the version-2 diagnostic exactly and does not turn
  its rejected K=4 fit into an accepted artifact. The integrated version-3
  fixed-K4 path passed 858 focused tests and a nonlaunching preflight. One full
  gate execution produced 2,543 passes, one stale golden subsequently fixed
  and targeted-pass verified, and one sandbox-blocked multiprocessing test;
  the exact blocked test then passed outside the sandbox. Aggregate
  branch-enabled coverage was 90.92%, Ruff was clean, and strict mypy passed.
  A representative real-HMM one-versus-nine test also passed.
- Noncanonical G3 rehearsal attempt 1 is retained as a legitimate resource
  non-pass because `rss_telemetry_complete=false`. After the bounded telemetry
  correction, attempt 2 traversed the closed 26-task graph, verified nine-worker
  parity for `hmm`, `kernel`, and `bridge`, and passed with no resource reasons.
  Its canonical-G3 projection is 82,591.74473571265 seconds, a
  9,149,177,600-byte peak, and 12,884,901,888 required free-disk bytes.
- Historical prelaunch checkpoint: canonical G3 was then unlaunched while
  scoped provenance, the final frozen gate/rehearsal/preflight, commit, and
  isolated worktree were completed. Those prerequisites later passed and the
  version-3 result is recorded above. No accepted model, generated return, or
  release exists.
- Version-4 smoke output and the current-source 100-LF/20-HF full-horizon pilot
  were byte-identical across representative worker counts. The full-horizon
  runs took 6.0406 seconds with one worker and 4.5798 seconds with nine.
- The nine-worker 200-LF/60-HF science pilot completed 1,082,000 rows in 42
  shards (93,284,667 bytes) in 7.9344 seconds. Streaming validation passed in
  1.3088 seconds with exact geometry, checksums, finite values, compounding,
  and zero duplicate complete paths. This remains pilot evidence; the complete
  production result is recorded separately above and in EV-045.
- The science pilot's daily reversal score was `+0.1367`, versus `-0.0835`
  for its unconditioned control (gap `+0.2201`) and `+0.0300` in actual history
  (actual-history-relative point gap `+0.1067`). Whole-month permutation reduced
  it to `+0.0309`, while within-month session permutation left it at `+0.1367`.
  Monthly whole-order permutation reduced quadratic alpha from `+0.2919` to
  `-0.028`; these findings are retained rather than treated as a production pass.
- The historical version-2 Phase-3 software rerun passed all 2,489 tests, Ruff,
  strict mypy, and every independent 80% coverage category; read-only G3
  preflight also passed without creating launch or scientific state.

[Unreleased]: https://example.invalid/prpg/compare/HEAD
