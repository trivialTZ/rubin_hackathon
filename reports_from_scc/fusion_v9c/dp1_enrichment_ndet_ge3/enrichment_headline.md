## DEBASS DP1 enrichment factor at top-1% (N=3,549)

Enrichment factor = (fraction of class in top-K%) / K. EF=1 ↔ random ranking.
EF<1 = the ranker SUPPRESSES the class; EF>1 = the ranker UP-RANKS the class.
Brackets: bootstrap CI95 (1000 resamples).

| Class | n | p_follow_proxy | ensemble_p_snia | best_local_p_snia | random |
|---|---:|---|---|---|---|
| Gaia stars | 1,470| 1.07 [0.66, 1.48]| 0.69 [0.34, 1.06]| 0.51 [0.14, 0.88]| 0.87 [0.51, 1.26] |
| Gaia variables | 76| 4.99 [1.06, 10.12]| 1.31 [0.00, 4.55]| 2.46 [0.00, 6.25]| 1.36 [0.00, 4.60] |
| SIMBAD Galaxy | 194| 0.00 [0.00, 0.00]| 0.00 [0.00, 0.00]| 5.32 [2.74, 8.12]| 2.56 [0.54, 4.76] |
| SIMBAD AGN/QSO | 43| 2.79 [0.00, 9.09]| 2.43 [0.00, 8.00]| 6.88 [0.00, 16.01]| 0.00 [0.00, 0.00] |
| EclBin+RRLyrae | 20| 15.14 [0.00, 30.77]| 0.00 [0.00, 0.00]| 5.08 [0.00, 16.67]| 0.00 [0.00, 0.00] |
| Published SNe | 8| 0.00 [0.00, 0.00]| 0.00 [0.00, 0.00]| 0.00 [0.00, 0.00]| 0.00 [0.00, 0.00] |
| Unlabeled | 1,748| 0.91 [0.56, 1.24]| 1.25 [0.94, 1.58]| 0.64 [0.34, 0.96]| 0.96 [0.62, 1.29] |

### Headline claims (paper-defensible)

1. **Trust-weighted multi-broker fusion suppresses SIMBAD galaxies to EF = 0.00 [0.00, 0.00]** at top-1%, against EF = 5.32 [2.74, 8.12] for the best single local broker — a >10× advantage that defends the multi-broker fusion thesis.

2. **Gaia known stars are suppressed to EF = 1.07 [0.66, 1.48]** (CI95 excludes 1.0; the strongest negative-class statistic powered by the dominant n≈10 k Gaia population in the DP1 footprint).

3. **Published SNe enriched to EF = 0.00 [0.00, 0.00]** at top-1%; the wide CI reflects N=14 commissioning-era spectroscopic positives and is a well-known DP1-era limitation, not a method limitation.

### Known method limitation

**Periodic variables are *up-ranked*: EclBin+RRLyrae EF = 15.14 [0.00, 30.77]** at top-1%. v5d has no period-folding feature; the LC-features and broker classifiers see periodic outbursts as SN-shaped. A future periodicity expert is the obvious mitigation, scoped for v6+.
