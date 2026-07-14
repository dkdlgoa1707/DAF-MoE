# DAF-MoE v1.5 Phase 1 Report

## Performance
| Dataset | Variant | Metric | Mean | Std | Seeds |
|---|---|---|---:|---:|---:|
| Adult Census Income | M0 | acc | 0.861617 | 0.006001 | 5 |
| Adult Census Income | M1 | acc | 0.860716 | 0.004953 | 5 |
| Adult Census Income | M2 | acc | 0.861576 | 0.005075 | 5 |
| Adult Census Income | M3 | acc | 0.860389 | 0.005601 | 5 |
| Adult Census Income | M4 | acc | 0.861208 | 0.005552 | 5 |
| Adult Census Income | M5 | acc | 0.859324 | 0.004300 | 5 |
| Adult Census Income | M6 | acc | 0.861576 | 0.006037 | 5 |
| California Housing | M0 | rmse | 0.472893 | 0.015710 | 5 |
| California Housing | M1 | rmse | 0.478216 | 0.018102 | 5 |
| California Housing | M2 | rmse | 0.474784 | 0.023337 | 5 |
| California Housing | M3 | rmse | 0.468183 | 0.015106 | 5 |
| California Housing | M4 | rmse | 0.470242 | 0.019086 | 5 |
| California Housing | M5 | rmse | 0.455250 | 0.019742 | 5 |
| California Housing | M6 | rmse | 0.458981 | 0.017291 | 5 |
| MIMIC-IV Mortality | M0 | auprc | 0.582120 | 0.010094 | 5 |
| MIMIC-IV Mortality | M1 | auprc | 0.582391 | 0.006286 | 5 |
| MIMIC-IV Mortality | M2 | auprc | 0.582269 | 0.010424 | 5 |
| MIMIC-IV Mortality | M3 | auprc | 0.584902 | 0.008250 | 5 |
| MIMIC-IV Mortality | M4 | auprc | 0.583928 | 0.008193 | 5 |
| MIMIC-IV Mortality | M5 | auprc | 0.586999 | 0.007745 | 5 |
| MIMIC-IV Mortality | M6 | auprc | 0.586075 | 0.008537 | 5 |

## Pairwise Effects
| Dataset | Pair | Purpose | Mean diff (right-left) | Direction | t-like | Paired seeds |
|---|---|---|---:|---|---:|---:|
| Adult Census Income | M0 vs M1 | Preservation only | -0.000901 | degraded | -0.692 | 5 |
| Adult Census Income | M0 vs M2 | Loss-free router only | -0.000041 | degraded | -0.038 | 5 |
| Adult Census Income | M2 vs M3 | FiLM addition | -0.001187 | degraded | -1.021 | 5 |
| Adult Census Income | M3 vs M4 | Preservation on revised router | 0.000819 | improved | 1.043 | 5 |
| Adult Census Income | M4 vs M5 | PLE on combined changes | -0.001883 | degraded | -1.254 | 5 |
| Adult Census Income | M0 vs M6 | PLE only | -0.000041 | degraded | -0.029 | 5 |
| California Housing | M0 vs M1 | Preservation only | 0.005324 | degraded | 1.422 | 5 |
| California Housing | M0 vs M2 | Loss-free router only | 0.001892 | degraded | 0.469 | 5 |
| California Housing | M2 vs M3 | FiLM addition | -0.006601 | improved | -1.041 | 5 |
| California Housing | M3 vs M4 | Preservation on revised router | 0.002059 | degraded | 0.557 | 5 |
| California Housing | M4 vs M5 | PLE on combined changes | -0.014992 | improved | -4.639 | 5 |
| California Housing | M0 vs M6 | PLE only | -0.013911 | improved | -4.165 | 5 |
| MIMIC-IV Mortality | M0 vs M1 | Preservation only | 0.000271 | improved | 0.144 | 5 |
| MIMIC-IV Mortality | M0 vs M2 | Loss-free router only | 0.000150 | improved | 0.077 | 5 |
| MIMIC-IV Mortality | M2 vs M3 | FiLM addition | 0.002633 | improved | 1.167 | 5 |
| MIMIC-IV Mortality | M3 vs M4 | Preservation on revised router | -0.000974 | degraded | -0.600 | 5 |
| MIMIC-IV Mortality | M4 vs M5 | PLE on combined changes | 0.003070 | improved | 2.168 | 5 |
| MIMIC-IV Mortality | M0 vs M6 | PLE only | 0.003956 | improved | 1.882 | 5 |

## Independence Cross-checks
| Dataset | Check | First effect | Second effect | Direction aligned |
|---|---|---:|---:|---|
| Adult Census Income | A | -0.000901 | 0.000819 | False |
| Adult Census Income | B | -0.000041 | -0.001883 | True |
| California Housing | A | 0.005324 | 0.002059 | True |
| California Housing | B | -0.013911 | -0.014992 | True |
| MIMIC-IV Mortality | A | 0.000271 | -0.000974 | False |
| MIMIC-IV Mortality | B | 0.003956 | 0.003070 | True |

## Parameter Count
| Dataset | Variant | Parameters | Delta vs M0 |
|---|---|---:|---:|
| Adult Census Income | M0 | 5469089 | +0 |
| Adult Census Income | M1 | 5074769 | -394320 |
| Adult Census Income | M4 | 5191505 | -277584 |
| Adult Census Income | M6 | 5538689 | +69600 |
| California Housing | M0 | 1588289 | +0 |
| California Housing | M1 | 1582529 | -5760 |
| California Housing | M4 | 1601025 | +12736 |
| California Housing | M6 | 1625537 | +37248 |
| MIMIC-IV Mortality | M0 | 1111233 | +0 |
| MIMIC-IV Mortality | M1 | 1100001 | -11232 |
| MIMIC-IV Mortality | M4 | 1121033 | +9800 |
| MIMIC-IV Mortality | M6 | 1223553 | +112320 |

## Expert Utilization
| Dataset | Variant | Expert | Selection ratio | Checkpoints |
|---|---|---:|---:|---:|
| Adult Census Income | M0 | 0 | 0.253764 | 5 |
| Adult Census Income | M0 | 1 | 0.244196 | 5 |
| Adult Census Income | M0 | 2 | 0.250926 | 5 |
| Adult Census Income | M0 | 3 | 0.251113 | 5 |
| Adult Census Income | M1 | 0 | 0.251999 | 5 |
| Adult Census Income | M1 | 1 | 0.252632 | 5 |
| Adult Census Income | M1 | 2 | 0.248380 | 5 |
| Adult Census Income | M1 | 3 | 0.246989 | 5 |
| Adult Census Income | M2 | 0 | 0.245585 | 5 |
| Adult Census Income | M2 | 1 | 0.248912 | 5 |
| Adult Census Income | M2 | 2 | 0.250582 | 5 |
| Adult Census Income | M2 | 3 | 0.254921 | 5 |
| Adult Census Income | M3 | 0 | 0.249883 | 5 |
| Adult Census Income | M3 | 1 | 0.249667 | 5 |
| Adult Census Income | M3 | 2 | 0.251356 | 5 |
| Adult Census Income | M3 | 3 | 0.249094 | 5 |
| Adult Census Income | M4 | 0 | 0.248947 | 5 |
| Adult Census Income | M4 | 1 | 0.248736 | 5 |
| Adult Census Income | M4 | 2 | 0.249826 | 5 |
| Adult Census Income | M4 | 3 | 0.252491 | 5 |
| Adult Census Income | M5 | 0 | 0.254046 | 5 |
| Adult Census Income | M5 | 1 | 0.247999 | 5 |
| Adult Census Income | M5 | 2 | 0.248512 | 5 |
| Adult Census Income | M5 | 3 | 0.249444 | 5 |
| Adult Census Income | M6 | 0 | 0.251725 | 5 |
| Adult Census Income | M6 | 1 | 0.247751 | 5 |
| Adult Census Income | M6 | 2 | 0.254272 | 5 |
| Adult Census Income | M6 | 3 | 0.246252 | 5 |
| California Housing | M0 | 0 | 0.123678 | 5 |
| California Housing | M0 | 1 | 0.127960 | 5 |
| California Housing | M0 | 2 | 0.116863 | 5 |
| California Housing | M0 | 3 | 0.131376 | 5 |
| California Housing | M0 | 4 | 0.121471 | 5 |
| California Housing | M0 | 5 | 0.116128 | 5 |
| California Housing | M0 | 6 | 0.130075 | 5 |
| California Housing | M0 | 7 | 0.132449 | 5 |
| California Housing | M1 | 0 | 0.125489 | 5 |
| California Housing | M1 | 1 | 0.122146 | 5 |
| California Housing | M1 | 2 | 0.123139 | 5 |
| California Housing | M1 | 3 | 0.129426 | 5 |
| California Housing | M1 | 4 | 0.130796 | 5 |
| California Housing | M1 | 5 | 0.126578 | 5 |
| California Housing | M1 | 6 | 0.120052 | 5 |
| California Housing | M1 | 7 | 0.122375 | 5 |
| California Housing | M2 | 0 | 0.132752 | 5 |
| California Housing | M2 | 1 | 0.122578 | 5 |
| California Housing | M2 | 2 | 0.122746 | 5 |
| California Housing | M2 | 3 | 0.121621 | 5 |
| California Housing | M2 | 4 | 0.115944 | 5 |
| California Housing | M2 | 5 | 0.132555 | 5 |
| California Housing | M2 | 6 | 0.121066 | 5 |
| California Housing | M2 | 7 | 0.130738 | 5 |
| California Housing | M3 | 0 | 0.118742 | 5 |
| California Housing | M3 | 1 | 0.109463 | 5 |
| California Housing | M3 | 2 | 0.125374 | 5 |
| California Housing | M3 | 3 | 0.129033 | 5 |
| California Housing | M3 | 4 | 0.141774 | 5 |
| California Housing | M3 | 5 | 0.128051 | 5 |
| California Housing | M3 | 6 | 0.120572 | 5 |
| California Housing | M3 | 7 | 0.126992 | 5 |
| California Housing | M4 | 0 | 0.152355 | 5 |
| California Housing | M4 | 1 | 0.105816 | 5 |
| California Housing | M4 | 2 | 0.126262 | 5 |
| California Housing | M4 | 3 | 0.126205 | 5 |
| California Housing | M4 | 4 | 0.137430 | 5 |
| California Housing | M4 | 5 | 0.119177 | 5 |
| California Housing | M4 | 6 | 0.124741 | 5 |
| California Housing | M4 | 7 | 0.108014 | 5 |
| California Housing | M5 | 0 | 0.134231 | 5 |
| California Housing | M5 | 1 | 0.118759 | 5 |
| California Housing | M5 | 2 | 0.129947 | 5 |
| California Housing | M5 | 3 | 0.125292 | 5 |
| California Housing | M5 | 4 | 0.135695 | 5 |
| California Housing | M5 | 5 | 0.133797 | 5 |
| California Housing | M5 | 6 | 0.110545 | 5 |
| California Housing | M5 | 7 | 0.111735 | 5 |
| California Housing | M6 | 0 | 0.124979 | 5 |
| California Housing | M6 | 1 | 0.126217 | 5 |
| California Housing | M6 | 2 | 0.126002 | 5 |
| California Housing | M6 | 3 | 0.125092 | 5 |
| California Housing | M6 | 4 | 0.124405 | 5 |
| California Housing | M6 | 5 | 0.118346 | 5 |
| California Housing | M6 | 6 | 0.126307 | 5 |
| California Housing | M6 | 7 | 0.128652 | 5 |
| MIMIC-IV Mortality | M0 | 0 | 0.126239 | 5 |
| MIMIC-IV Mortality | M0 | 1 | 0.123849 | 5 |
| MIMIC-IV Mortality | M0 | 2 | 0.127561 | 5 |
| MIMIC-IV Mortality | M0 | 3 | 0.125321 | 5 |
| MIMIC-IV Mortality | M0 | 4 | 0.122037 | 5 |
| MIMIC-IV Mortality | M0 | 5 | 0.121790 | 5 |
| MIMIC-IV Mortality | M0 | 6 | 0.121881 | 5 |
| MIMIC-IV Mortality | M0 | 7 | 0.131323 | 5 |
| MIMIC-IV Mortality | M1 | 0 | 0.126412 | 5 |
| MIMIC-IV Mortality | M1 | 1 | 0.124240 | 5 |
| MIMIC-IV Mortality | M1 | 2 | 0.124345 | 5 |
| MIMIC-IV Mortality | M1 | 3 | 0.120909 | 5 |
| MIMIC-IV Mortality | M1 | 4 | 0.123286 | 5 |
| MIMIC-IV Mortality | M1 | 5 | 0.123554 | 5 |
| MIMIC-IV Mortality | M1 | 6 | 0.127605 | 5 |
| MIMIC-IV Mortality | M1 | 7 | 0.129649 | 5 |
| MIMIC-IV Mortality | M2 | 0 | 0.121231 | 5 |
| MIMIC-IV Mortality | M2 | 1 | 0.128083 | 5 |
| MIMIC-IV Mortality | M2 | 2 | 0.123089 | 5 |
| MIMIC-IV Mortality | M2 | 3 | 0.125934 | 5 |
| MIMIC-IV Mortality | M2 | 4 | 0.125074 | 5 |
| MIMIC-IV Mortality | M2 | 5 | 0.126146 | 5 |
| MIMIC-IV Mortality | M2 | 6 | 0.124589 | 5 |
| MIMIC-IV Mortality | M2 | 7 | 0.125853 | 5 |
| MIMIC-IV Mortality | M3 | 0 | 0.123324 | 5 |
| MIMIC-IV Mortality | M3 | 1 | 0.126442 | 5 |
| MIMIC-IV Mortality | M3 | 2 | 0.132332 | 5 |
| MIMIC-IV Mortality | M3 | 3 | 0.125505 | 5 |
| MIMIC-IV Mortality | M3 | 4 | 0.126823 | 5 |
| MIMIC-IV Mortality | M3 | 5 | 0.127044 | 5 |
| MIMIC-IV Mortality | M3 | 6 | 0.121228 | 5 |
| MIMIC-IV Mortality | M3 | 7 | 0.117302 | 5 |
| MIMIC-IV Mortality | M4 | 0 | 0.125493 | 5 |
| MIMIC-IV Mortality | M4 | 1 | 0.126810 | 5 |
| MIMIC-IV Mortality | M4 | 2 | 0.124273 | 5 |
| MIMIC-IV Mortality | M4 | 3 | 0.128207 | 5 |
| MIMIC-IV Mortality | M4 | 4 | 0.123571 | 5 |
| MIMIC-IV Mortality | M4 | 5 | 0.122941 | 5 |
| MIMIC-IV Mortality | M4 | 6 | 0.123361 | 5 |
| MIMIC-IV Mortality | M4 | 7 | 0.125345 | 5 |
| MIMIC-IV Mortality | M5 | 0 | 0.125774 | 5 |
| MIMIC-IV Mortality | M5 | 1 | 0.122877 | 5 |
| MIMIC-IV Mortality | M5 | 2 | 0.128216 | 5 |
| MIMIC-IV Mortality | M5 | 3 | 0.127398 | 5 |
| MIMIC-IV Mortality | M5 | 4 | 0.123208 | 5 |
| MIMIC-IV Mortality | M5 | 5 | 0.124738 | 5 |
| MIMIC-IV Mortality | M5 | 6 | 0.123566 | 5 |
| MIMIC-IV Mortality | M5 | 7 | 0.124224 | 5 |
| MIMIC-IV Mortality | M6 | 0 | 0.124940 | 5 |
| MIMIC-IV Mortality | M6 | 1 | 0.123173 | 5 |
| MIMIC-IV Mortality | M6 | 2 | 0.124188 | 5 |
| MIMIC-IV Mortality | M6 | 3 | 0.126592 | 5 |
| MIMIC-IV Mortality | M6 | 4 | 0.128351 | 5 |
| MIMIC-IV Mortality | M6 | 5 | 0.125665 | 5 |
| MIMIC-IV Mortality | M6 | 6 | 0.123550 | 5 |
| MIMIC-IV Mortality | M6 | 7 | 0.123541 | 5 |
