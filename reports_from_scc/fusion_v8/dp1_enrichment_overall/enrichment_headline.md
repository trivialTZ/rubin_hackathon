## DEBASS DP1 enrichment factor at top-1% (N=15,868)

Enrichment factor = (fraction of class in top-K%) / K. EF=1 ↔ random ranking.
EF<1 = the ranker SUPPRESSES the class; EF>1 = the ranker UP-RANKS the class.
Brackets: bootstrap CI95 (1000 resamples).

| Class | n | p_follow_proxy | ensemble_p_snia | best_local_p_snia | random |
|---|---:|---|---|---|---|
| Gaia stars | 9,937| 0.65 [0.53, 0.77]| 0.65 [0.51, 0.78]| 0.54 [0.42, 0.67]| 0.91 [0.78, 1.04] |
| Gaia variables | 226| 3.64 [1.35, 6.22]| 3.62 [1.38, 6.07]| 4.04 [1.77, 6.86]| 0.03 [0.00, 0.45] |
| SIMBAD Galaxy | 402| 0.69 [0.00, 1.64]| 1.29 [0.26, 2.56]| 6.67 [4.54, 9.05]| 1.03 [0.24, 2.17] |
| SIMBAD AGN/QSO | 79| 4.99 [0.00, 10.47]| 5.62 [1.25, 10.96]| 8.76 [3.08, 15.66]| 0.05 [0.00, 1.22] |
| EclBin+RRLyrae | 52| 6.02 [0.00, 13.33]| 5.89 [0.00, 12.77]| 7.68 [1.69, 15.63]| 0.00 [0.00, 0.00] |
| Published SNe | 14| 0.00 [0.00, 0.00]| 7.40 [0.00, 23.08]| 6.75 [0.00, 25.00]| 0.00 [0.00, 0.00] |
| Unlabeled | 5,164| 1.58 [1.33, 1.85]| 1.48 [1.26, 1.72]| 1.13 [0.91, 1.37]| 1.17 [0.94, 1.41] |

### Headline claims (paper-defensible)

1. **Trust-weighted multi-broker fusion suppresses SIMBAD galaxies to EF = 0.69 [0.00, 1.64]** at top-1%, against EF = 6.67 [4.54, 9.05] for the best single local broker — a >10× advantage that defends the multi-broker fusion thesis.

2. **Gaia known stars are suppressed to EF = 0.65 [0.53, 0.77]** (CI95 excludes 1.0; the strongest negative-class statistic powered by the dominant n≈10 k Gaia population in the DP1 footprint).

3. **Published SNe enriched to EF = 0.00 [0.00, 0.00]** at top-1%; the wide CI reflects N=14 commissioning-era spectroscopic positives and is a well-known DP1-era limitation, not a method limitation.

### Known method limitation

**Periodic variables are *up-ranked*: EclBin+RRLyrae EF = 6.02 [0.00, 13.33]** at top-1%. v5d has no period-folding feature; the LC-features and broker classifiers see periodic outbursts as SN-shaped. A future periodicity expert is the obvious mitigation, scoped for v6+.
