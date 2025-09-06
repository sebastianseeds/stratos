# Dose Calculation Uncertainty Methodology

## Overview

The STRATOS dose calculator implements uncertainty estimation for radiation dose calculations along orbital trajectories. Error bands in the plots represent 1-sigma confidence intervals.

## Instantaneous Dose Rate Uncertainties

Dose rate uncertainties are constant in time. The model combines five components using root-sum-of-squares:

### 1. Poisson Statistical Uncertainty
- Relative uncertainty: 1/√N_eff where N_eff is effective particle count
- Typical: 1-10% for moderate fluxes

### 2. Flux Field Interpolation Uncertainty  
- Relative uncertainty: 10%
- From VTK spatial interpolation between grid points

### 3. Cross-Sectional Area Uncertainty
- Relative uncertainty: 2%
- From geometric measurement precision

### 4. Energy Conversion Factor Uncertainty
- Relative uncertainty: 5%
- From particle energy spectrum and stopping power uncertainties

### 5. Baseline Uncertainty
- Absolute uncertainty: 1% of maximum dose rate
- Represents measurement noise floor

## Total Instantaneous Uncertainty

```
σ_rate = √(σ_poisson² + σ_interp² + σ_area² + σ_energy² + σ_baseline²)
```

This uncertainty remains constant throughout the measurement period.

## Cumulative Dose Uncertainty Propagation

Cumulative dose uncertainties grow with time due to error accumulation through integration.

### Statistical Error Propagation

Integration of uncertain measurements:
```
σ²_statistical(t) = Σᵢ [σ_rate(tᵢ) × Δtᵢ]²
```
Grows as √N where N is the number of measurement intervals.

### Systematic Drift

Calibration drift and model uncertainties:
```
σ_systematic(t) = cumulative_dose(t) × 0.002 × t_hours
```
0.2% per hour drift from calibration and environmental changes.

### Correlation Factor

Accounts for persistent systematic biases:
```
correlation_factor = 1 + 0.001 × √t_hours
```

### Total Cumulative Uncertainty

```
σ_cumulative = √[(σ_statistical × correlation_factor)² + σ_systematic²]
```

## Typical Uncertainty Magnitudes

| Flux Region | Dose Rate | Cumulative Dose (1h) | Cumulative Dose (24h) |
|-------------|-----------|---------------------|----------------------|
| Van Allen Belt Peak | 8-15% | 10-20% | 20-40% |
| Moderate Flux | 12-20% | 15-25% | 25-50% |
| Low Flux | 20-50% | 25-60% | 40-80% |

## Implementation

Error bands are displayed using `matplotlib.fill_between()` to show continuous uncertainty regions rather than discrete error bars.

Methods: `DoseWindow._calculate_dose_rate_errors()` and `DoseWindow._calculate_cumulative_dose_errors()` in `/flux_visualizer/ui/windows/dose_window.py`