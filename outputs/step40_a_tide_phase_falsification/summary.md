# Step 40 Tide Phase Falsification

Run command: `python src/step40_a_tide_phase_falsification.py`

## Fair alignment comparison

| Phase generator | Mean corr | Median corr |
| --- | --- | --- |
| clock_24h | 0.483 | 0.578 |
| solar_semidiurnal_12h | 0.032 | 0.065 |
| M2_12.4206h | 0.060 | 0.134 |
| N2_12.6583h | 0.129 | 0.221 |
| K1_23.9345h | 0.483 | 0.574 |
| clock_folded_12h | 0.455 | 0.504 |
| random_day_phase (500x) | 0.483 | — |

## Key comparisons
- M2 - Clock: -0.423
- M2 - Solar12: +0.027
- M2 - Folded12: -0.395

## Deployable templates (target_like)


## Decision
- M2 - Clock >= 0.10: NO (-0.423)
- M2 - Folded12 >= 0.05: NO (-0.395)
- M2 > Solar12: YES (diff +0.027)