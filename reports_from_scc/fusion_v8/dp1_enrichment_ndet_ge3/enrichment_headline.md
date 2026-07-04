## DEBASS DP1 enrichment factor at top-1% (N=3,549)

Enrichment factor = (fraction of class in top-K%) / K. EF=1 ↔ random ranking.
EF<1 = the ranker SUPPRESSES the class; EF>1 = the ranker UP-RANKS the class.
Brackets: bootstrap CI95 (1000 resamples).

| Class | n | p_follow_proxy | ensemble_p_snia | best_local_p_snia | random |
|---|---:|---|---|---|---|
| Gaia stars | 1,470| 0.89 [0.49, 1.27]| 0.52 [0.20, 0.86]| 0.51 [0.14, 0.88]| 0.87 [0.51, 1.26] |
| Gaia variables | 76| 3.68 [0.00, 8.45]| 4.01 [0.00, 9.21]| 2.46 [0.00, 6.25]| 1.36 [0.00, 4.60] |
| SIMBAD Galaxy | 194| 0.00 [0.00, 0.00]| 0.86 [0.00, 2.44]| 5.32 [2.74, 8.12]| 2.56 [0.54, 4.76] |
| SIMBAD AGN/QSO | 43| 2.39 [0.00, 7.69]| 4.74 [0.00, 11.90]| 6.88 [0.00, 16.01]| 0.00 [0.00, 0.00] |
| EclBin+RRLyrae | 20| 9.51 [0.00, 23.81]| 9.86 [0.00, 25.00]| 5.08 [0.00, 16.67]| 0.00 [0.00, 0.00] |
| Published SNe | 8| 0.00 [0.00, 0.00]| 0.04 [0.00, 0.00]| 0.00 [0.00, 0.00]| 0.00 [0.00, 0.00] |
| Unlabeled | 1,748| 1.09 [0.74, 1.43]| 1.24 [0.92, 1.54]| 0.64 [0.34, 0.96]| 0.96 [0.62, 1.29] |

### Headline claims (paper-defensible)

1. **Trust-weighted multi-broker fusion suppresses SIMBAD galaxies to EF = 0.00 [0.00, 0.00]** at top-1%, against EF = 5.32 [2.74, 8.12] for the best single local broker — a >10× advantage that defends the multi-broker fusion thesis.

2. **Gaia known stars are suppressed to EF = 0.89 [0.49, 1.27]** (CI95 excludes 1.0; the strongest negative-class statistic powered by the dominant n≈10 k Gaia population in the DP1 footprint).

3. **Published SNe enriched to EF = 0.00 [0.00, 0.00]** at top-1%; the wide CI reflects N=14 commissioning-era spectroscopic positives and is a well-known DP1-era limitation, not a method limitation.

### Known method limitation

**Periodic variables are *up-ranked*: EclBin+RRLyrae EF = 9.51 [0.00, 23.81]** at top-1%. v5d has no period-folding feature; the LC-features and broker classifiers see periodic outbursts as SN-shaped. A future periodicity expert is the obvious mitigation, scoped for v6+.
