# Changelog

All notable changes to BindsNET are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/). For releases prior to the entries below,
see the [GitHub releases / tags](https://github.com/BindsNET/bindsnet/releases).

## [Unreleased]

### Added
- Sparse-tensor support for additional learning rules (plus a batch dimension and docs
  for `sparse=True`).
- Validation tests for the reward-modulated learning rules `MSTDP` and `MSTDPET`.
- Regression test for a preallocated `Monitor` short-run bug (PR #761).
- Read the Docs configuration for documentation builds.
- Reproducibility/transparency docs: `DATA.md` (dataset & stimulus declaration),
  `REPRODUCING.md` (model→script→command→seed map), `CITATION.cff`, and a
  `docs/source/models_spec.rst` neural-model specification page.

### Changed
- `assign_labels` / evaluation: handle abstention for inactive samples, mark
  never-firing neurons with `-1`, and accuracy/performance improvements.
- CI: dropped Python 3.10 (project requires `>=3.11`); upgraded GitHub Actions; test on
  Python 3.11/3.12/3.13. README Python requirement aligned to `>=3.11,<3.14`.
- Routine dependency updates via Poetry.

### Fixed
- `bernoulli_loader` now honors the `max_prob` kwarg (PR #743).
- Bug with preallocated buffers and `torch.cat`.
- `torch.save` compatibility for PyTorch 2.6.0.
- Python 3.13 support / tests.
- `eth_mnist` example.

## [0.3.3] - 2024-10-18

Baseline for this changelog. See the
[releases page](https://github.com/BindsNET/bindsnet/releases) for the history of
0.1.x–0.3.3.
