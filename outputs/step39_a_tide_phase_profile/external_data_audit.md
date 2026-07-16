# External Data Audit

## Competition rules
The competition description states predictions are based on given AIS training data.
It does not explicitly prohibit or permit external data.
Top 10 submissions require source code review.

## Tide data
Source: Simplified astronomical tide prediction using published harmonic constants
for Tianjin Xingang (117.79°E, 38.97°N). Uses M2, S2, K1, O1, N2 constituents.
This is purely astronomical prediction — no observed water level data used.
Astronomical tide is deterministic and computable from celestial mechanics,
available at any future date without real-time observation.

## Data type: predicted_astronomical_tide
- NOT observed water level
- NOT weather data
- NOT port operation records
- Computed from first principles (celestial mechanics)

## Availability at prediction time
Astronomical tide predictions for 2018-01-25..31 were computable on 2018-01-24.
No future information leakage.

## Caveat
Harmonic constants are approximate. Exact tide timing may differ by 15-30 minutes.
This step tests the HYPOTHESIS of tidal phase driving, not precise tide prediction.
