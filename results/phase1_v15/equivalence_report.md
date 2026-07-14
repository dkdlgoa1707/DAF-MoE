# DAF-MoE v1.5 Equivalence Report

Seed: 42; tolerance: atol=1e-05, rtol=0.0001

| Dataset | Scenario A strategy | A pass | A max diff | B pass | B changed diff | B restored diff |
|---|---|---:|---:|---:|---:|---:|
| california | normal | True | 0 | True | 0.0019029826 | 0 |
| adult | random | True | 0 | True | 0.00062765181 | 0 |
| mimic4 | normal | True | 0 | True | 0.00031331182 | 0 |

Scenario A requires full flag-off equivalence for a non-linspace strategy.
Scenario B requires natural linspace outputs to differ and mu-restored outputs to match.

Overall: PASS
