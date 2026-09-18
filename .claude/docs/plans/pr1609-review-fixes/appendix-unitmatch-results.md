Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (n=10); 20 units/session, |S|=5 per seed -> pooled S true pairs = 50, non-S true pairs = 150

## Table A: true pairs (i,i) pooled over seeds. pass = both directions > 0.5 (backend rule); mean_prob = mean of the two directions

| scenario | cond | S: pass/n | S: mean prob | S: min prob | nonS: pass/n | nonS: mean prob | nonS: min prob | argmax hits S/n | argmax hits nonS/n |
|---|---|---|---|---|---|---|---|---|---|
| control | current | 0/0 | nan | nan | 178/200 | 0.905 | 0.000 | 0/0 | 178/200 |
| control | fixed | 0/0 | nan | nan | 174/200 | 0.896 | 0.000 | 0/0 | 174/200 |
| driftout_A | current | 0/50 | 0.465 | 0.000 | 133/150 | 0.911 | 0.000 | 0/50 | 133/150 |
| driftout_A | fixed | 42/50 | 0.870 | 0.000 | 130/150 | 0.905 | 0.000 | 42/50 | 130/150 |
| driftout_AB | current | 0/50 | 0.000 | 0.000 | 132/150 | 0.922 | 0.007 | 0/50 | 132/150 |
| driftout_AB | fixed | 45/50 | 0.905 | 0.000 | 127/150 | 0.899 | 0.000 | 45/50 | 127/150 |
| zeroed_A | current | 0/50 | 0.458 | 0.000 | 134/150 | 0.907 | 0.001 | 0/50 | 134/150 |
| zeroed_AB | current | 0/50 | 0.000 | 0.000 | 134/150 | 0.922 | 0.003 | 0/50 | 133/150 |

## Table B: false cross-session pairs (i != j) pooled. n pass (both dirs > 0.5) / n pairs, and max mean-prob

| scenario | cond | SxS pass/n (max) | SxnonS pass/n (max) | nonSxnonS pass/n (max) |
|---|---|---|---|---|
| control | current | 0/0 (nan) | 0/0 (nan) | 15/3800 (0.999) |
| control | fixed | 0/0 (nan) | 0/0 (nan) | 25/3800 (0.998) |
| driftout_A | current | 0/200 (0.267) | 2/1500 (0.864) | 10/2100 (0.962) |
| driftout_A | fixed | 2/200 (0.794) | 9/1500 (0.998) | 17/2100 (0.996) |
| driftout_AB | current | 0/200 (0.000) | 0/1500 (0.492) | 21/2100 (0.999) |
| driftout_AB | fixed | 2/200 (0.786) | 7/1500 (0.998) | 18/2100 (0.996) |
| zeroed_A | current | 0/200 (0.303) | 0/1500 (0.515) | 5/2100 (0.966) |
| zeroed_AB | current | 0/200 (0.000) | 0/1500 (0.491) | 28/2100 (0.995) |

## Table C: collateral on NON-S pairs, paired against the same seed's control run of the same condition.

zeroed_* vs control/current: non-S templates bit-identical -> pure calibration effect of zero halves.
driftout_* vs control/fixed: non-S templates bit-identical -> effect of S units' changed (but non-zero) templates.
driftout_* vs control/current: CONFOUNDED (non-S half-2 random subsample differs), shown for completeness.

| comparison | nonS true: mean dprob | max |dprob| | pass ctrl -> scen | flips pass->fail / fail->pass | nonSxnonS false: pass ctrl -> scen | max |dprob| | seeds with any nonS true flip |
|---|---|---|---|---|---|---|---|
| zeroed_A/current vs control/current | 0.006 | 0.163 | 133 -> 134 | 0 / 1 | 10 -> 5 | 0.904 | 1/10 |
| zeroed_AB/current vs control/current | 0.021 | 0.599 | 133 -> 134 | 0 / 1 | 10 -> 28 | 0.974 | 1/10 |
| driftout_A/fixed vs control/fixed | -0.001 | 0.151 | 132 -> 130 | 2 / 0 | 17 -> 17 | 0.612 | 2/10 |
| driftout_AB/fixed vs control/fixed | -0.007 | 0.401 | 132 -> 127 | 5 / 0 | 17 -> 18 | 0.400 | 4/10 |
| driftout_A/current vs control/current | 0.010 | 0.376 | 133 -> 133 | 1 / 1 | 10 -> 10 | 0.587 | 2/10 |
| driftout_AB/current vs control/current | 0.020 | 0.591 | 133 -> 132 | 3 / 2 | 10 -> 21 | 0.976 | 5/10 |

## Table D: calibration state, mean over seeds (n_expected_matches feeds the prior; thr = get_threshold pass 2; diag = within-session self total_score cv0 vs cv1)

| scenario | cond | n_expected_matches | prior(match) | thr pass2 | cand diag (of 40) | diag A_S | diag A_nonS | diag B_S | diag B_nonS | |drift| max |
|---|---|---|---|---|---|---|---|---|---|---|
| control | current | 86.4 | 0.054 | 0.705 | 37.3 | nan | 0.864 | nan | 0.875 | 0.025 |
| control | fixed | 90.0 | 0.056 | 0.716 | 37.4 | nan | 0.872 | nan | 0.873 | 0.048 |
| driftout_A | current | 73.3 | 0.046 | 0.740 | 32.4 | 0.061 | 0.879 | 0.881 | 0.889 | 0.082 |
| driftout_A | fixed | 89.8 | 0.056 | 0.719 | 37.4 | 0.872 | 0.876 | 0.863 | 0.875 | 0.044 |
| driftout_AB | current | 67.1 | 0.042 | 0.718 | 28.2 | 0.063 | 0.877 | 0.093 | 0.890 | 0.141 |
| driftout_AB | fixed | 89.6 | 0.056 | 0.711 | 37.4 | 0.873 | 0.876 | 0.877 | 0.874 | 0.048 |
| zeroed_A | current | 69.4 | 0.043 | 0.744 | 32.4 | 0.060 | 0.880 | 0.879 | 0.887 | 0.032 |
| zeroed_AB | current | 70.6 | 0.044 | 0.708 | 28.3 | 0.064 | 0.876 | 0.092 | 0.883 | 0.024 |

## Table E: naive-Bayes kernel means E[score|cond], mean over seeds (c1 = candidate-match class, c0 = non-match)

| scenario | cond | amp c1/c0 | spatial_decay c1/c0 | centroid_overlord c1/c0 | centroid_dist c1/c0 | waveform c1/c0 | trajectory c1/c0 |
|---|---|---|---|---|---|---|---|
| control | current | 0.84/0.39 | 0.82/0.37 | 0.88/0.71 | 0.87/0.51 | 0.75/0.47 | 0.77/0.57 |
| control | fixed | 0.81/0.39 | 0.81/0.38 | 0.88/0.71 | 0.84/0.51 | 0.75/0.49 | 0.76/0.56 |
| driftout_A | current | 0.84/0.36 | 0.82/0.37 | 0.86/0.65 | 0.87/0.47 | 0.76/0.42 | 0.76/0.60 |
| driftout_A | fixed | 0.82/0.39 | 0.81/0.38 | 0.88/0.72 | 0.85/0.51 | 0.76/0.49 | 0.77/0.56 |
| driftout_AB | current | 0.82/0.29 | 0.81/0.34 | 0.87/0.50 | 0.85/0.39 | 0.72/0.32 | 0.75/0.64 |
| driftout_AB | fixed | 0.83/0.39 | 0.82/0.37 | 0.88/0.73 | 0.85/0.51 | 0.75/0.48 | 0.77/0.56 |
| zeroed_A | current | 0.84/0.36 | 0.83/0.37 | 0.87/0.65 | 0.85/0.47 | 0.76/0.42 | 0.77/0.60 |
| zeroed_AB | current | 0.81/0.29 | 0.81/0.34 | 0.86/0.50 | 0.84/0.39 | 0.73/0.33 | 0.75/0.65 |

## Table F: directional probabilities involving a zero half (CURRENT). prob[A_i,B_j] = A cv0 vs B cv1; prob[B_j,A_i] = B cv0 vs A cv1

| scenario | pair type | n | A->B mean/max | B->A mean/max | n both>0.5 | n either>0.5 |
|---|---|---|---|---|---|---|
| driftout_A | true S (i,i) | 50 | 0.930/1.000 | 0.000/0.000 | 0 | 47 |
| driftout_A | false SxS (i!=j) | 200 | 0.006/0.533 | 0.000/0.000 | 0 | 1 |
| driftout_AB | true S (i,i) | 50 | 0.000/0.000 | 0.000/0.000 | 0 | 0 |
| driftout_AB | false SxS (i!=j) | 200 | 0.000/0.000 | 0.000/0.000 | 0 | 0 |
| zeroed_A | true S (i,i) | 50 | 0.917/1.000 | 0.000/0.000 | 0 | 46 |
| zeroed_A | false SxS (i!=j) | 200 | 0.004/0.606 | 0.000/0.000 | 0 | 1 |
| zeroed_AB | true S (i,i) | 50 | 0.000/0.000 | 0.000/0.000 | 0 | 0 |
| zeroed_AB | false SxS (i!=j) | 200 | 0.000/0.000 | 0.000/0.000 | 0 | 0 |

## Table G: per-seed S true-pair pass counts (of 5) and non-S (of 15)

| seed | S | dA cur S | dA fix S | dAB cur S | dAB fix S | ctrl cur nonS(/20) | ctrl fix nonS(/20) | dA cur nonS | dA fix nonS | zA cur nonS | dAB cur nonS | dAB fix nonS | zAB cur nonS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | [5, 6, 9, 10, 13] | 0 | 5 | 0 | 4 | 17 | 16 | 12 | 10 | 12 | 11 | 10 | 12 |
| 1 | [0, 7, 8, 13, 18] | 0 | 4 | 0 | 5 | 18 | 18 | 13 | 14 | 13 | 14 | 14 | 13 |
| 2 | [1, 4, 5, 8, 13] | 0 | 5 | 0 | 5 | 18 | 18 | 13 | 13 | 13 | 13 | 13 | 13 |
| 3 | [1, 3, 4, 12, 19] | 0 | 5 | 0 | 5 | 19 | 19 | 13 | 15 | 14 | 13 | 15 | 14 |
| 4 | [9, 11, 15, 16, 18] | 0 | 3 | 0 | 3 | 19 | 17 | 15 | 14 | 15 | 15 | 14 | 15 |
| 5 | [0, 9, 10, 13, 15] | 0 | 4 | 0 | 4 | 16 | 18 | 13 | 14 | 13 | 13 | 13 | 12 |
| 6 | [6, 7, 9, 17, 18] | 0 | 4 | 0 | 5 | 19 | 18 | 15 | 14 | 15 | 14 | 12 | 15 |
| 7 | [10, 11, 12, 15, 17] | 0 | 3 | 0 | 5 | 19 | 16 | 14 | 12 | 14 | 14 | 12 | 14 |
| 8 | [3, 4, 5, 11, 18] | 0 | 4 | 0 | 4 | 16 | 18 | 13 | 14 | 13 | 13 | 14 | 13 |
| 9 | [2, 5, 6, 14, 17] | 0 | 5 | 0 | 5 | 17 | 16 | 12 | 10 | 12 | 12 | 10 | 13 |

## Timings, mean per scenario (s)

| scenario | current bundle | current match | fixed bundle | fixed match |
|---|---|---|---|---|
| control | 0.41 | 0.11 | 0.32 | 0.11 |
| driftout_A | 0.33 | 0.11 | 0.32 | 0.11 |
| driftout_AB | 0.32 | 0.10 | 0.33 | 0.11 |

## Sanity checks

- CURRENT bundle A half 1 all-zero exactly for S in every drift-out run: True
- CURRENT bundle B half 0 all-zero exactly for S in every driftout_AB run: True
- FIXED bundles have no all-zero halves in any run: True
- Replicated inference == UnitMatchBackend.match (same pair set, identical probabilities) in every run: True
- Control (S empty) true-pair pass counts per seed, CURRENT vs FIXED: [(17, 16), (18, 18), (18, 18), (19, 19), (19, 17), (16, 18), (19, 18), (19, 16), (16, 18), (17, 16)]
