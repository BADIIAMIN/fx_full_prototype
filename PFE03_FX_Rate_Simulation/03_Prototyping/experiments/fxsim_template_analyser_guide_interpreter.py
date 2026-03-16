"""
FX MODEL VALIDATION - ANALYSIS TEMPLATE
Purpose: Guide for interpreting test results and documenting findings
"""

# ============================================================================
# ANALYSIS TEMPLATE FOR DISTRIBUTION TESTS
# ============================================================================

DISTRIBUTION_ANALYSIS_TEMPLATE = """
## Test 1A: Jarque-Bera Test for Normality

### Test Objective
Assess whether standardized FX returns follow a normal distribution under the 
constant-volatility GBM assumption.

### Results
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| JB Statistic | {jb_stat:.2f} | N/A | - |
| P-Value | {jb_pval:.4f} | > 0.05 (pass) | {jb_conclusion} |
| Severity | {jb_severity} | - | - |

### Interpretation
{jb_interpretation}

### Implications for PFE Model
{jb_pfe_impact}

### Recommendation
{jb_recommendation}

---

## Test 1B: Kurtosis Analysis

### Test Objective
Quantify tail thickness relative to normal distribution. Excess kurtosis > 0 
indicates fatter tails than Gaussian, which can lead to PFE underestimation.

### Results
| Metric | Value | Acceptable Range | Status |
|--------|-------|------------------|--------|
| Excess Kurtosis | {kurt_value:.3f} | -0.5 to 1.0 | {kurt_conclusion} |
| Severity | {kurt_severity} | - | - |

### Interpretation
{kurt_interpretation}

### Implications for PFE Model
{kurt_pfe_impact}

### Recommendation
{kurt_recommendation}

---

## Test 1C: Anderson-Darling Test (Tail-Sensitive)

### Test Objective
Test normality with emphasis on tail regions, which are critical for PFE₉₅/PFE₉₉.

### Results
| Metric | Value | Threshold (5%) | Status |
|--------|-------|----------------|--------|
| AD Statistic | {ad_stat:.3f} | {ad_crit:.3f} | {ad_conclusion} |
| Severity | {ad_severity} | - | - |

### Interpretation
{ad_interpretation}

### Implications for PFE Model
{ad_pfe_impact}

### Recommendation
{ad_recommendation}

---

## Test 1D: Q-Q Plot Analysis

### Test Objective
Visually and quantitatively assess tail deviations from theoretical normal distribution.

### Results
| Quantile | Theoretical | Empirical | Deviation | % Deviation |
|----------|-------------|-----------|-----------|-------------|
| 1st | {q01_theo:.3f} | {q01_emp:.3f} | {q01_dev:.3f} | {q01_pct:.1f}% |
| 99th | {q99_theo:.3f} | {q99_emp:.3f} | {q99_dev:.3f} | {q99_pct:.1f}% |

**Acceptance Criteria:**
- 5th/95th percentile: < 0.3 deviation ✓
- 1st/99th percentile: < 0.5 deviation ✓
- Status: {qq_conclusion} ({qq_severity})

### Interpretation
{qq_interpretation}

### Implications for PFE Model
{qq_pfe_impact}

### Recommendation
{qq_recommendation}

---

## DISTRIBUTION TESTS: OVERALL ASSESSMENT

### Summary Decision Matrix

| Test | Status | Severity | Weight | Weighted Score |
|------|--------|----------|--------|----------------|
| Jarque-Bera | {jb_conclusion} | {jb_severity} | 20% | {jb_score} |
| Kurtosis | {kurt_conclusion} | {kurt_severity} | 30% | {kurt_score} |
| Anderson-Darling | {ad_conclusion} | {ad_severity} | 30% | {ad_score} |
| Q-Q Plot | {qq_conclusion} | {qq_severity} | 20% | {qq_score} |
| **OVERALL** | **{overall_conclusion}** | **{overall_severity}** | 100% | **{total_score}** |

### Overall Finding Classification

**Risk Rating:** {risk_rating}

{overall_finding_text}

### Recommended Actions

{action_plan}

### Monitoring Requirements

{monitoring_plan}
"""

# ============================================================================
# ANALYSIS TEMPLATE FOR PIT TESTS
# ============================================================================

PIT_ANALYSIS_TEMPLATE = """
## Test 2A: Kolmogorov-Smirnov Uniformity Test

### Test Objective
Validate that Probability Integral Transform (PIT) values are uniformly distributed,
indicating correct model calibration.

### Results
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| KS Statistic | {ks_stat:.4f} | < critical | - |
| P-Value | {ks_pval:.4f} | > 0.05 | {ks_conclusion} |
| Severity | {ks_severity} | - | - |

### Interpretation
{ks_interpretation}

### Calibration Implications
{ks_calib_impact}

### Recommendation
{ks_recommendation}

---

## Test 2B: PIT Histogram Analysis

### Test Objective
Visual and statistical check for uniform distribution of PIT values.
Non-uniform patterns indicate calibration bias or volatility clustering.

### Results
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Chi-Square | {chi2_stat:.2f} | - | - |
| P-Value | {chi2_pval:.4f} | > 0.05 | {chi2_conclusion} |
| Severity | {chi2_severity} | - | - |

### Interpretation
{chi2_interpretation}

### Calibration Implications
{chi2_calib_impact}

### Recommendation
{chi2_recommendation}

---

## Test 2C: Kupiec Exceedance Test (95th Percentile)

### Test Objective
Test if 95th percentile is breached with expected frequency (5% of time).

### Results
| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Observed Exceedances | {kupiec95_obs} | {kupiec95_exp:.1f} | - |
| Exceedance Rate | {kupiec95_rate:.2%} | 5.00% | - |
| LR Statistic | {kupiec95_lr:.3f} | - | - |
| P-Value | {kupiec95_pval:.4f} | > 0.05 | {kupiec95_conclusion} |
| Severity | {kupiec95_severity} | - | - |

### Interpretation
{kupiec95_interpretation}

### PFE₉₅ Implications
{kupiec95_pfe_impact}

### Recommendation
{kupiec95_recommendation}

---

## Test 2D: Kupiec Exceedance Test (99th Percentile)

### Test Objective
Test if 99th percentile is breached with expected frequency (1% of time).

### Results
| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Observed Exceedances | {kupiec99_obs} | {kupiec99_exp:.1f} | - |
| Exceedance Rate | {kupiec99_rate:.2%} | 1.00% | - |
| LR Statistic | {kupiec99_lr:.3f} | - | - |
| P-Value | {kupiec99_pval:.4f} | > 0.05 | {kupiec99_conclusion} |
| Severity | {kupiec99_severity} | - | - |

### Interpretation
{kupiec99_interpretation}

### PFE₉₉ Implications
{kupiec99_pfe_impact}

### Recommendation
{kupiec99_recommendation}

---

## PIT TESTS: OVERALL CALIBRATION ASSESSMENT

### Summary

**Calibration Quality:** {calib_quality}
**Risk Rating:** {calib_risk_rating}

{calib_overall_text}

### Recommended Actions

{calib_action_plan}
"""

# ============================================================================
# ANALYSIS TEMPLATE FOR TAIL RISK VALIDATION
# ============================================================================

TAIL_RISK_ANALYSIS_TEMPLATE = """
## Test 3A: Historical Exceedance Analysis - 95th Percentile

### Test Objective
Compare realized 95th percentile vs model prediction over rolling 1-year windows.

### Results
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean Ratio (Realized/Model) | {q95_mean:.3f} | < 1.20 | {q95_conclusion} |
| Median Ratio | {q95_median:.3f} | - | - |
| Std Dev of Ratios | {q95_std:.3f} | - | - |
| % Periods > 1.2 | {q95_pct_over:.1f}% | < 20% | {q95_threshold_breach} |
| Severity | {q95_severity} | - | - |

### Interpretation
{q95_interpretation}

### PFE₉₅ Impact Assessment
{q95_pfe_impact}

### Recommendation
{q95_recommendation}

---

## Test 3B: Historical Exceedance Analysis - 99th Percentile

### Test Objective
Compare realized 99th percentile vs model prediction over rolling 1-year windows.

### Results
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Mean Ratio (Realized/Model) | {q99_mean:.3f} | < 1.20 | {q99_conclusion} |
| Median Ratio | {q99_median:.3f} | - | - |
| Std Dev of Ratios | {q99_std:.3f} | - | - |
| % Periods > 1.2 | {q99_pct_over:.1f}% | < 20% | {q99_threshold_breach} |
| Severity | {q99_severity} | - | - |

### Interpretation
{q99_interpretation}

### PFE₉₉ Impact Assessment
{q99_pfe_impact}

### Recommendation
{q99_recommendation}

---

## TAIL RISK: OVERALL ASSESSMENT

### Coverage Quality
{tail_overall_quality}

### Risk Rating
{tail_risk_rating}

### Materiality for PFE
{tail_materiality}

### Action Plan
{tail_action_plan}
"""

# ============================================================================
# ANALYSIS TEMPLATE FOR STRESS PERIOD VALIDATION
# ============================================================================

STRESS_ANALYSIS_TEMPLATE = """
## Test 4: Stress Period Coverage Analysis

### Test Objective
Verify model PFE₉₉ (30-day horizon) covers realized maximum moves during 
known stress events.

### Stress Period 1: COVID-19 Crisis (Feb-Apr 2020)

| Metric | Value | Status |
|--------|-------|--------|
| Realized Max Move | {covid_realized:.2%} | - |
| Model PFE₉₉ (30d) | {covid_model:.2%} | - |
| Coverage Ratio | {covid_ratio:.2f}x | {covid_coverage} |
| Severity | {covid_severity} | - |

**Interpretation:** {covid_interpretation}

---

### Stress Period 2: Brexit (Jun-Jul 2016)

| Metric | Value | Status |
|--------|-------|--------|
| Realized Max Move | {brexit_realized:.2%} | - |
| Model PFE₉₉ (30d) | {brexit_model:.2%} | - |
| Coverage Ratio | {brexit_ratio:.2f}x | {brexit_coverage} |
| Severity | {brexit_severity} | - |

**Interpretation:** {brexit_interpretation}

---

### Stress Period 3: Russia-Ukraine (Feb-Mar 2022)

| Metric | Value | Status |
|--------|-------|--------|
| Realized Max Move | {ukraine_realized:.2%} | - |
| Model PFE₉₉ (30d) | {ukraine_model:.2%} | - |
| Coverage Ratio | {ukraine_ratio:.2f}x | {ukraine_coverage} |
| Severity | {ukraine_severity} | - |

**Interpretation:** {ukraine_interpretation}

---

## STRESS TEST: OVERALL ASSESSMENT

### Coverage Summary

**Total Stress Events Tested:** {total_events}
**Events Covered:** {events_covered}
**Coverage Rate:** {coverage_rate:.0%}

**Acceptance Criterion:** ≥ 95% coverage (allow 1 breach out of 3)
**Result:** {stress_overall_conclusion}

### Risk Rating
{stress_risk_rating}

### Implications
{stress_implications}

### Recommended Actions
{stress_action_plan}

---

## REMEDIATION HIERARCHY

Based on stress test results:

### If Coverage ≥ 95% (PASS):
1. ✅ Accept model for production use
2. ✅ Implement standard monitoring
3. ✅ Annual re-validation cycle

### If Coverage 66-94% (CONDITIONAL PASS):
1. ⚠️ Implement stressed volatility overlay (σ_stress = 1.5 × σ_calib)
2. ⚠️ Monthly monitoring of vol regime indicators
3. ⚠️ Quarterly stress test refreshes

### If Coverage < 66% (FAIL):
1. 🔴 Do NOT approve for production
2. 🔴 Require model enhancement:
   - Option A: EWMA volatility (adaptive)
   - Option B: Regime-switching model
   - Option C: Stochastic volatility (Heston)
3. 🔴 Re-validation after enhancement
"""

# ============================================================================
# MASTER REPORT GENERATOR
# ============================================================================

def generate_validation_analysis_report(results: Dict, fx_pair: str = "EURUSD") -> str:
    """
    Generate formatted analysis report from test results
    """
    
    # Extract results
    dist_results = results['distribution_tests']
    pit_results = results['pit_tests']
    tail_results = results['tail_risk']
    stress_results = results['stress_coverage']
    
    # Format distribution analysis
    dist_analysis = DISTRIBUTION_ANALYSIS_TEMPLATE.format(
        # Jarque-Bera
        jb_stat=dist_results.loc[dist_results['Test']=='Jarque-Bera', 'Statistic/Value'].values[0],
        jb_pval=0.0234,  # Example - extract from actual results
        jb_conclusion="FAIL",
        jb_severity="MEDIUM",
        jb_interpretation="Returns exhibit non-normality (p=0.0234 < 0.05)",
        jb_pfe_impact="May underestimate tail exposures; consider vol overlay",
        jb_recommendation="Implement quarterly kurtosis monitoring with ±2 threshold",
        
        # Kurtosis
        kurt_value=1.85,
        kurt_conclusion="PASS",
        kurt_severity="LOW",
        kurt_interpretation="Moderate fat tails but within acceptable range",
        kurt_pfe_impact="Tail risk moderately higher than Gaussian; monitor",
        kurt_recommendation="Acceptable; continue with baseline model + monitoring",
        
        # Add all other parameters...
        # (continuing with complete mapping)
        
        # Overall
        overall_conclusion="CONDITIONAL PASS",
        overall_severity="MEDIUM",
        risk_rating="AMBER",
        overall_finding_text="Model exhibits moderate deviations from normality...",
        action_plan="1. Implement vol overlay\n2. Quarterly monitoring\n3. Annual review",
        monitoring_plan="Track kurtosis monthly; escalate if excess > 2.0"
    )
    
    # Combine all sections
    full_report = f"""
{'='*80}
FX SIMULATION MODEL - VALIDATION ANALYSIS REPORT
Currency Pair: {fx_pair}
Validation Date: 2026-02-09
{'='*80}

{dist_analysis}

{PIT_ANALYSIS_TEMPLATE}

{TAIL_RISK_ANALYSIS_TEMPLATE}

{STRESS_ANALYSIS_TEMPLATE}

{'='*80}
END OF REPORT
{'='*80}
"""
    
    return full_report


# ============================================================================
# DECISION MATRIX HELPER
# ============================================================================

def compute_overall_risk_rating(dist_severity, pit_severity, tail_severity, stress_severity):
    """
    Aggregate test severities into overall risk rating
    """
    severity_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
    
    scores = [
        severity_map[dist_severity] * 0.25,   # 25% weight
        severity_map[pit_severity] * 0.20,    # 20% weight
        severity_map[tail_severity] * 0.30,   # 30% weight
        severity_map[stress_severity] * 0.25  # 25% weight
    ]
    
    total_score = sum(scores)
    
    if total_score <= 1.5:
        return "GREEN", "PASS", "Model approved for production with standard monitoring"
    elif total_score <= 2.3:
        return "AMBER", "CONDITIONAL PASS", "Approved with enhanced monitoring and vol overlay"
    else:
        return "RED", "FAIL", "Model requires enhancement before production use"


# Example interpretation helper
def interpret_kurtosis_for_finding(excess_kurtosis: float) -> Dict:
    """
    Generate Finding 01 text based on kurtosis result
    """
    if excess_kurtosis > 2.0:
        return {
            'severity': 'HIGH',
            'finding_text': f"""
Finding 01: Material Fat Tails Detected (Excess Kurtosis = {excess_kurtosis:.2f})

The constant-volatility GBM assumption is materially violated. Observed excess 
kurtosis of {excess_kurtosis:.2f} indicates returns have significantly fatter 
tails than the normal distribution implied by the model.

Impact on PFE:
- PFE₉₉ may be underestimated by 15-30%
- Stress scenarios systematically under-represent extreme moves
- Path-dependent products (barriers, TARFs) have biased exposures

Recommendations:
1. MANDATORY: Implement stressed vol multiplier (×1.5 for PFE₉₉)
2. Enhance model with stochastic volatility or jump-diffusion
3. Monthly kurtosis monitoring with auto-escalation if >2.5
4. Portfolio-level impact assessment on XCCY swaps required
            """,
            'action_required': True
        }
    
    elif 1.0 < excess_kurtosis <= 2.0:
        return {
            'severity': 'MEDIUM',
            'finding_text': f"""
Observation: Moderate Fat Tails (Excess Kurtosis = {excess_kurtosis:.2f})

Returns exhibit moderate tail thickness above Gaussian. While within tolerance,
this suggests GBM may underestimate extreme quantiles during stress.

Recommendations:
1. Implement quarterly kurtosis monitoring
2. Sensitivity test: run PFE with σ_stressed = 1.25 × σ_baseline
3. Document limitation in model use policy
            """,
            'action_required': False
        }
    
    else:
        return {
            'severity': 'LOW',
            'finding_text': f"""
No Material Concerns (Excess Kurtosis = {excess_kurtosis:.2f})

Tail behavior consistent with constant-vol GBM assumption.
Standard monitoring procedures apply.
            """,
            'action_required': False
        }