# Upstream and attribution

A100Supermimic is an independent, unofficial derivative of [RealMimic](https://github.com/kevinhenryzz/realmimic), pinned at `61f2973f6426255a8c844d6ca0c894eff4d8f45c`. The upstream project is Apache-2.0 and includes an MIT-licensed Supermimic_mod training core; see LICENSE, NOTICE, and LICENSES. The original upstream citation, including its DOI and authorship, is retained in [docs/REALMIMIC_UPSTREAM_CITATION.cff](docs/REALMIMIC_UPSTREAM_CITATION.cff); it identifies RealMimic, not a new A100Supermimic release.

Our work adds the A100/CUDA 12 environment lock, resumable training, timing, strict coverage/flag reporting, data selections and experimental adaptive-horizon infrastructure. The adaptive method is off in the baseline and has no demonstrated gain. The A100 full-library seed-0 training is our run; its first raw evaluation remains invalid because of simulator flags, while a subsequent full re-evaluation with explicitly different solver settings passed the strict flag check. See README for status and scope.

The upstream README is retained at `docs/REALMIMIC_UPSTREAM_README.md` for attribution. Its reported GPU results, site material, videos and released weights are the upstream author's, not our results. Third-party data, assets and weights have separate terms.

The initial A100Supermimic public repository is a source snapshot of the above upstream revision plus our modifications, with its own Git history. The upstream demo website and its bundled media are intentionally omitted; the original upstream project remains available at the link above.
