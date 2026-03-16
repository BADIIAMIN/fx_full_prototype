"""
FX MODEL VALIDATION - 5-DAY RETURN CALIBRATION ANALYSIS
=========================================================
Purpose: Validate the assumption that σ_daily = σ_5day / √5 and assess
         the appropriateness of 5-day calibration for PFE modeling

Tests Included:
1. Volatility Scaling Comparison (√5 rule validation)
2. Autocorrelation Test (i.i.d. assumption)
3. ARCH Effects Test (volatility clustering)
4. GARCH Model Comparison (time-varying volatility)
5. PFE Impact Analysis (1d vs 5d calibration)
6. MPOR Alignment Test (10-day margin period)
7. Short-Term Spike Smoothing Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# MAIN VALIDATION CLASS
# ============================================================================

class FXCalibrationValidator:
    """
    Comprehensive validator for 5-day vs 1-day return calibration methodology
    """
    
    def __init__(self, fx_data: pd.Series, name: str = "EURUSD"):
        """
        Parameters:
        -----------
        fx_data : pd.Series
            Daily FX spot rate time series with DatetimeIndex
        name : str
            Currency pair name for reporting
        """
        self.fx_data = fx_data
        self.name = name
        self.returns_1d = None
        self.returns_5d = None
        self.returns_5d_overlap = None
        self.results = {}
        
    def compute_returns(self):
        """
        Compute both 1-day and 5-day log returns
        """
        # 1-day returns
        self.returns_1d = np.log(self.fx_data / self.fx_data.shift(1)).dropna()
        
        # 5-day returns (non-overlapping for calibration)
        self.returns_5d = np.log(self.fx_data / self.fx_data.shift(5)).dropna()[::5]
        
        # 5-day returns (overlapping for robustness checks)
        self.returns_5d_overlap = np.log(self.fx_data / self.fx_data.shift(5)).dropna()
        
        print(f"Returns computed:")
        print(f"  1-day returns: {len(self.returns_1d)} observations")
        print(f"  5-day returns (non-overlapping): {len(self.returns_5d)} observations")
        print(f"  5-day returns (overlapping): {len(self.returns_5d_overlap)} observations")
        
        return self.returns_1d, self.returns_5d
    
    # ========================================================================
    # TEST 1: VOLATILITY SCALING COMPARISON
    # ========================================================================
    
    def volatility_comparison(self) -> Dict:
        """
        Test the square-root-of-time scaling rule: σ(5d) = σ(1d) × √5
        
        Returns:
        --------
        Dict with volatility estimates and scaling validation results
        """
        if self.returns_1d is None:
            self.compute_returns()
        
        # Method 1: Direct from 1-day returns
        sigma_1d_direct = self.returns_1d.std()
        sigma_annual_from_1d = sigma_1d_direct * np.sqrt(252)
        
        # Method 2: From 5-day returns, scaled down
        sigma_5d = self.returns_5d.std()
        sigma_1d_from_5d = sigma_5d / np.sqrt(5)
        sigma_annual_from_5d = sigma_1d_from_5d * np.sqrt(252)
        
        # Method 3: Overlapping 5-day (for comparison)
        sigma_5d_overlap = self.returns_5d_overlap.std()
        sigma_1d_from_5d_overlap = sigma_5d_overlap / np.sqrt(5)
        
        # Test theoretical relationship: σ(5d) should equal σ(1d) × √5
        theoretical_sigma_5d = sigma_1d_direct * np.sqrt(5)
        scaling_ratio = sigma_5d / theoretical_sigma_5d
        deviation_pct = (scaling_ratio - 1.0) * 100
        
        # Assess pass/fail
        if abs(deviation_pct) < 5:
            conclusion = 'PASS'
            severity = 'LOW'
        elif abs(deviation_pct) < 10:
            conclusion = 'BORDERLINE'
            severity = 'MEDIUM'
        else:
            conclusion = 'FAIL'
            severity = 'HIGH'
        
        result = {
            'test_name': 'Volatility Scaling Comparison',
            'sigma_1d_direct': sigma_1d_direct,
            'sigma_1d_from_5d': sigma_1d_from_5d,
            'sigma_5d_actual': sigma_5d,
            'sigma_5d_theoretical': theoretical_sigma_5d,
            'scaling_ratio': scaling_ratio,
            'deviation_pct': deviation_pct,
            'annual_vol_from_1d': sigma_annual_from_1d,
            'annual_vol_from_5d': sigma_annual_from_5d,
            'annual_vol_difference': abs(sigma_annual_from_5d - sigma_annual_from_1d),
            'conclusion': conclusion,
            'severity': severity
        }
        
        self.results['volatility_scaling'] = result
        return result
    
    # ========================================================================
    # TEST 2: AUTOCORRELATION TEST
    # ========================================================================
    
    def autocorrelation_test(self, lags: int = 20) -> Dict:
        """
        Test for autocorrelation in returns using Ljung-Box test
        Null hypothesis: No autocorrelation (returns are i.i.d.)
        
        Parameters:
        -----------
        lags : int
            Number of lags to test (default 20)
        
        Returns:
        --------
        Dict with Ljung-Box test results
        """
        if self.returns_1d is None:
            self.compute_returns()
        
        # Ljung-Box test
        lb_result = acorr_ljungbox(self.returns_1d, lags=lags, return_df=True)
        
        # Count significant lags
        significant_lags = lb_result[lb_result['lb_pvalue'] < 0.05]
        pct_significant = len(significant_lags) / lags * 100
        min_pvalue = lb_result['lb_pvalue'].min()
        
        # Assess severity
        if pct_significant < 10:
            conclusion = 'PASS'
            severity = 'LOW'
        elif pct_significant < 25:
            conclusion = 'BORDERLINE FAIL'
            severity = 'MEDIUM'
        else:
            conclusion = 'FAIL'
            severity = 'HIGH'
        
        result = {
            'test_name': 'Ljung-Box Autocorrelation Test',
            'lags_tested': lags,
            'significant_lags': len(significant_lags),
            'pct_significant': pct_significant,
            'min_p_value': min_pvalue,
            'lb_statistic_max': lb_result['lb_stat'].max(),
            'conclusion': conclusion,
            'severity': severity,
            'detailed_results': lb_result
        }
        
        self.results['autocorrelation'] = result
        return result
    
    # ========================================================================
    # TEST 3: ARCH EFFECTS TEST
    # ========================================================================
    
    def arch_effect_test(self, lags: int = 5) -> Dict:
        """
        Test for ARCH effects (volatility clustering) using Engle's ARCH-LM test
        Null hypothesis: No ARCH effects (homoskedastic residuals)
        
        Parameters:
        -----------
        lags : int
            Number of lags for ARCH test (default 5)
        
        Returns:
        --------
        Dict with ARCH-LM test results
        """
        if self.returns_1d is None:
            self.compute_returns()
        
        # ARCH-LM test
        arch_test = het_arch(self.returns_1d, nlags=lags)
        lm_stat, lm_pvalue, f_stat, f_pvalue = arch_test
        
        # Test on squared returns (direct volatility clustering test)
        returns_squared = self.returns_1d ** 2
        lb_squared = acorr_ljungbox(returns_squared, lags=10, return_df=True)
        
        # Assess severity
        if lm_pvalue > 0.05:
            conclusion = 'PASS'
            severity = 'LOW'
        elif lm_pvalue > 0.01:
            conclusion = 'FAIL'
            severity = 'MEDIUM'
        else:
            conclusion = 'FAIL'
            severity = 'HIGH'
        
        result = {
            'test_name': f'ARCH-LM Test (lag={lags})',
            'lm_statistic': lm_stat,
            'lm_p_value': lm_pvalue,
            'f_statistic': f_stat,
            'f_p_value': f_pvalue,
            'squared_returns_lb_min_pvalue': lb_squared['lb_pvalue'].min(),
            'conclusion': conclusion,
            'severity': severity
        }
        
        self.results['arch_effects'] = result
        return result
    
    # ========================================================================
    # TEST 4: GARCH MODEL COMPARISON
    # ========================================================================
    
    def garch_model_comparison(self) -> Dict:
        """
        Fit GARCH(1,1) model to assess time-varying volatility
        Compare unconditional GARCH volatility to constant calibration
        
        Returns:
        --------
        Dict with GARCH model results and persistence analysis
        """
        if self.returns_1d is None:
            self.compute_returns()
        
        # Scale returns to percentage for numerical stability
        returns_pct = self.returns_1d * 100
        
        try:
            # Fit GARCH(1,1)
            garch = arch_model(returns_pct, vol='Garch', p=1, q=1, rescale=False)
            garch_fit = garch.fit(disp='off', show_warning=False)
            
            # Extract parameters
            omega = garch_fit.params['omega']
            alpha = garch_fit.params['alpha[1]']
            beta = garch_fit.params['beta[1]']
            
            # Persistence
            persistence = alpha + beta
            
            # Unconditional volatility
            uncond_var = omega / (1 - alpha - beta)
            uncond_vol_daily = np.sqrt(uncond_var) / 100  # Back to decimal
            uncond_vol_annual = uncond_vol_daily * np.sqrt(252)
            
            # Compare to constant vol
            constant_vol_annual = self.returns_1d.std() * np.sqrt(252)
            
            # Conditional volatility series
            cond_vol = garch_fit.conditional_volatility / 100
            
            # Assess persistence severity
            if persistence < 0.85:
                severity = 'LOW'
                conclusion = 'MODERATE'
            elif persistence < 0.95:
                severity = 'MEDIUM'
                conclusion = 'SIGNIFICANT'
            else:
                severity = 'HIGH'
                conclusion = 'VERY_SIGNIFICANT'
            
            result = {
                'test_name': 'GARCH(1,1) Model',
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'persistence': persistence,
                'uncond_vol_annual': uncond_vol_annual,
                'constant_vol_annual': constant_vol_annual,
                'vol_difference_pct': (uncond_vol_annual - constant_vol_annual) / constant_vol_annual * 100,
                'aic': garch_fit.aic,
                'bic': garch_fit.bic,
                'conditional_vol_series': cond_vol,
                'conclusion': conclusion,
                'severity': severity
            }
            
        except Exception as e:
            result = {
                'test_name': 'GARCH(1,1) Model',
                'error': str(e),
                'conclusion': 'ERROR',
                'severity': 'N/A'
            }
        
        self.results['garch'] = result
        return result
    
    # ========================================================================
    # TEST 5: PFE IMPACT ANALYSIS
    # ========================================================================
    
    def pfe_impact_analysis(self, horizon_days: int = 10, quantile: float = 0.99) -> Dict:
        """
        Assess impact of 1d vs 5d calibration on PFE calculations
        
        Parameters:
        -----------
        horizon_days : int
            PFE horizon in business days (default 10)
        quantile : float
            Risk quantile (default 0.99 for 99th percentile)
        
        Returns:
        --------
        Dict with PFE comparison and dollar impact
        """
        if self.returns_1d is None:
            self.compute_returns()
        
        # Volatility from each method
        sigma_1d_direct = self.returns_1d.std()
        sigma_5d = self.returns_5d.std()
        sigma_1d_from_5d = sigma_5d / np.sqrt(5)
        
        # PFE calculations
        z_score = stats.norm.ppf(quantile)
        dt = horizon_days / 252
        
        pfe_from_1d = z_score * sigma_1d_direct * np.sqrt(dt)
        pfe_from_5d = z_score * sigma_1d_from_5d * np.sqrt(dt)
        
        # Difference
        pfe_diff_pct = (pfe_from_5d - pfe_from_1d) / pfe_from_1d * 100
        
        # Dollar impact (assume $1B notional)
        notional = 1_000_000_000
        pfe_dollar_from_1d = pfe_from_1d * notional
        pfe_dollar_from_5d = pfe_from_5d * notional
        pfe_dollar_diff = pfe_dollar_from_5d - pfe_dollar_from_1d
        
        # Assess materiality
        if abs(pfe_diff_pct) < 3:
            conclusion = 'PASS'
            severity = 'LOW'
        elif abs(pfe_diff_pct) < 7:
            conclusion = 'BORDERLINE'
            severity = 'MEDIUM'
        else:
            conclusion = 'FAIL'
            severity = 'HIGH'
        
        result = {
            'test_name': f'PFE Impact Analysis ({horizon_days}d horizon, Q{int(quantile*100)})',
            'sigma_1d_direct': sigma_1d_direct,
            'sigma_1d_from_5d': sigma_1d_from_5d,
            'pfe_from_1d_pct': pfe_from_1d * 100,
            'pfe_from_5d_pct': pfe_from_5d * 100,
            'pfe_difference_pct': pfe_diff_pct,
            'pfe_from_1d_dollar': pfe_dollar_from_1d,
            'pfe_from_5d_dollar': pfe_dollar_from_5d,
            'pfe_difference_dollar': pfe_dollar_diff,
            'conclusion': conclusion,
            'severity': severity
        }
        
        self.results['pfe_impact'] = result
        return result
    
    # ========================================================================
    # TEST 6: MPOR ALIGNMENT TEST
    # ========================================================================
    
    def mpor_alignment_test(self) -> Dict:
        """
        Test alignment with Margin Period of Risk (MPOR)
        Standard MPOR: 10 business days for non-centrally cleared derivatives
        
        Returns:
        --------
        Dict with cross-frequency volatility stability analysis
        """
        if self.returns_1d is None:
            self.compute_returns()
        
        # Compute volatility from different return frequencies
        return_frequencies = {
            '1-day': (1, self.returns_1d),
            '5-day': (5, self.returns_5d),
            '10-day': (10, np.log(self.fx_data / self.fx_data.shift(10)).dropna()[::10]),
            '20-day': (20, np.log(self.fx_data / self.fx_data.shift(20)).dropna()[::20])
        }
        
        results_by_freq = {}
        annual_vols = []
        
        for freq_name, (days, returns) in return_frequencies.items():
            sigma_raw = returns.std()
            sigma_daily_scaled = sigma_raw / np.sqrt(days)
            sigma_annual = sigma_daily_scaled * np.sqrt(252)
            
            results_by_freq[freq_name] = {
                'days': days,
                'sigma_raw': sigma_raw,
                'sigma_daily_scaled': sigma_daily_scaled,
                'sigma_annual': sigma_annual,
                'n_observations': len(returns)
            }
            annual_vols.append(sigma_annual)
        
        # Assess stability
        vol_range = max(annual_vols) - min(annual_vols)
        vol_mean = np.mean(annual_vols)
        vol_cv = vol_range / vol_mean  # Coefficient of variation
        
        # MPOR-specific (10-day)
        sigma_10d = results_by_freq['10-day']['sigma_annual']
        sigma_1d = results_by_freq['1-day']['sigma_annual']
        mpor_alignment_pct = abs(sigma_10d - sigma_1d) / sigma_1d * 100
        
        # Assess
        if vol_cv < 0.05 and mpor_alignment_pct < 5:
            conclusion = 'PASS'
            severity = 'LOW'
        elif vol_cv < 0.10 and mpor_alignment_pct < 10:
            conclusion = 'PASS'
            severity = 'MEDIUM'
        else:
            conclusion = 'FAIL'
            severity = 'HIGH'
        
        result = {
            'test_name': 'MPOR Alignment Test',
            'frequency_results': results_by_freq,
            'vol_range': vol_range,
            'vol_mean': vol_mean,
            'vol_cv_pct': vol_cv * 100,
            'mpor_10d_alignment_pct': mpor_alignment_pct,
            'conclusion': conclusion,
            'severity': severity
        }
        
        self.results['mpor_alignment'] = result
        return result
    
    # ========================================================================
    # TEST 7: SHORT-TERM SPIKE SMOOTHING
    # ========================================================================
    
    def short_term_spike_analysis(self) -> Dict:
        """
        Assess whether 5-day calibration smooths short-term volatility spikes
        
        Returns:
        --------
        Dict with spike detection and capture rate analysis
        """
        if self.returns_1d is None:
            self.compute_returns()
        
        # Identify spikes in 1-day returns (>3σ)
        abs_returns_1d = np.abs(self.returns_1d)
        mean_abs = abs_returns_1d.mean()
        std_abs = abs_returns_1d.std()
        threshold_3sigma = mean_abs + 3 * std_abs
        
        spikes_1d = abs_returns_1d[abs_returns_1d > threshold_3sigma]
        n_spikes = len(spikes_1d)
        pct_spikes = n_spikes / len(abs_returns_1d) * 100
        
        # Check if spikes are captured in 5-day windows
        spike_dates = spikes_1d.index
        captures = []
        
        for spike_date in spike_dates:
            try:
                # Find 5-day return window containing this spike
                window_start_idx = max(0, self.fx_data.index.get_loc(spike_date) - 5)
                window_start = self.fx_data.index[window_start_idx]
                
                # Get 5-day return
                fx_start = self.fx_data.loc[window_start]
                fx_end = self.fx_data.loc[spike_date]
                return_5d = np.log(fx_end / fx_start)
                
                # Would this trigger in 5-day calibration?
                sigma_5d = self.returns_5d.std()
                threshold_5d = 3 * sigma_5d
                
                captured = abs(return_5d) > threshold_5d
                captures.append(captured)
            except:
                pass
        
        capture_rate = np.mean(captures) * 100 if captures else 0
        smoothing_effect = 100 - capture_rate
        
        # Assess severity
        if capture_rate > 70:
            conclusion = 'MODERATE_SMOOTHING'
            severity = 'LOW'
        elif capture_rate > 40:
            conclusion = 'SIGNIFICANT_SMOOTHING'
            severity = 'MEDIUM'
        else:
            conclusion = 'SEVERE_SMOOTHING'
            severity = 'HIGH'
        
        result = {
            'test_name': 'Short-Term Spike Smoothing Analysis',
            'n_spikes_1d': n_spikes,
            'pct_observations_with_spikes': pct_spikes,
            'avg_spike_magnitude': spikes_1d.mean(),
            'max_spike_magnitude': spikes_1d.max(),
            'spike_capture_rate_5d': capture_rate,
            'smoothing_effect_pct': smoothing_effect,
            'conclusion': conclusion,
            'severity': severity
        }
        
        self.results['spike_smoothing'] = result
        return result
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_volatility_clustering(self, save_path: str = None):
        """
        Create comprehensive visualization of volatility clustering patterns
        """
        if self.returns_1d is None:
            self.compute_returns()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'{self.name}: Volatility Clustering Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Returns time series
        axes[0, 0].plot(self.returns_1d.index, self.returns_1d.values, 
                        linewidth=0.8, alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Daily Returns', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Absolute returns
        axes[0, 1].plot(self.returns_1d.index, np.abs(self.returns_1d.values), 
                        linewidth=0.8, alpha=0.7, color='darkred')
        axes[0, 1].set_title('Absolute Returns (Volatility Proxy)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('|Return|')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling volatility (20-day)
        rolling_vol_1d = self.returns_1d.rolling(20).std() * np.sqrt(252)
        axes[1, 0].plot(rolling_vol_1d.index, rolling_vol_1d.values, 
                        linewidth=1.5, color='purple', label='20-day Rolling Vol')
        axes[1, 0].axhline(self.returns_1d.std() * np.sqrt(252), 
                           color='green', linestyle='--', linewidth=2, 
                           label='Full Sample Vol')
        axes[1, 0].set_title('Rolling Volatility (20-day)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Annualized Vol')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ACF of absolute returns
        plot_acf(np.abs(self.returns_1d), lags=20, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title('ACF of Absolute Returns', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. ACF of squared returns
        plot_acf(self.returns_1d ** 2, lags=20, ax=axes[2, 0], alpha=0.05)
        axes[2, 0].set_title('ACF of Squared Returns', 
                            fontsize=12, fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 1d vs 5d calibration comparison
        rolling_vol_5d = self.returns_5d_overlap.rolling(20).std() / np.sqrt(5) * np.sqrt(252)
        
        common_idx = rolling_vol_1d.index.intersection(rolling_vol_5d.index)
        if len(common_idx) > 0:
            axes[2, 1].plot(common_idx, rolling_vol_1d.loc[common_idx].values, 
                            linewidth=1.5, color='blue', label='From 1-day returns', alpha=0.7)
            axes[2, 1].plot(common_idx, rolling_vol_5d.loc[common_idx].values, 
                            linewidth=1.5, color='orange', label='From 5-day returns', alpha=0.7)
        axes[2, 1].set_title('Rolling Vol: 1-day vs 5-day Calibration', 
                            fontsize=12, fontweight='bold')
        axes[2, 1].set_ylabel('Annualized Vol')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all_calibration_tests(self) -> pd.DataFrame:
        """
        Execute complete calibration validation test suite
        
        Returns:
        --------
        DataFrame with summary of all test results
        """
        print(f"\n{'='*80}")
        print(f"5-DAY CALIBRATION VALIDATION: {self.name}")
        print(f"{'='*80}\n")
        
        # Compute returns
        self.compute_returns()
        print()
        
        # Run all tests
        tests = [
            ("Volatility Scaling Comparison", self.volatility_comparison),
            ("Autocorrelation Test", self.autocorrelation_test),
            ("ARCH Effects Test", self.arch_effect_test),
            ("GARCH Model Comparison", self.garch_model_comparison),
            ("PFE Impact Analysis", self.pfe_impact_analysis),
            ("MPOR Alignment Test", self.mpor_alignment_test),
            ("Short-Term Spike Analysis", self.short_term_spike_analysis)
        ]
        
        for i, (test_name, test_func) in enumerate(tests, 1):
            print(f"Test {i}/7: {test_name}...")
            test_func()
        
        print("\nGenerating Visualizations...")
        self.plot_volatility_clustering()
        
        # Create summary table
        summary_data = []
        for test_name, test_result in self.results.items():
            if 'test_name' in test_result:
                summary_data.append({
                    'Test': test_result['test_name'],
                    'Conclusion': test_result.get('conclusion', 'N/A'),
                    'Severity': test_result.get('severity', 'N/A')
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"\n{'='*80}")
        print("CALIBRATION VALIDATION COMPLETE")
        print(f"{'='*80}\n")
        
        return summary_df
    
    # ========================================================================
    # DETAILED RESULTS PRINTER
    # ========================================================================
    
    def print_detailed_results(self):
        """
        Print detailed results for all tests
        """
        print(f"\n{'='*80}")
        print("DETAILED TEST RESULTS")
        print(f"{'='*80}\n")
        
        for test_name, results in self.results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            print("-" * 80)
            for key, value in results.items():
                if key not in ['detailed_results', 'frequency_results', 
                              'conditional_vol_series']:
                    if isinstance(value, float):
                        if abs(value) < 0.01:
                            print(f"  {key:.<45} {value:.6f}")
                        else:
                            print(f"  {key:.<45} {value:.4f}")
                    else:
                        print(f"  {key:.<45} {value}")


# ============================================================================
# EXAMPLE USAGE WITH REALISTIC DATA
# ============================================================================

def generate_realistic_fx_data(n_days: int = 1510, start_date: str = '2020-01-01'):
    """
    Generate realistic FX data with GARCH volatility clustering
    
    Parameters:
    -----------
    n_days : int
        Number of trading days to generate
    start_date : str
        Start date for the series
    
    Returns:
    --------
    pd.Series with FX rates
    """
    np.random.seed(42)
    dates = pd.date_range(start_date, periods=n_days, freq='B')
    
    # GARCH parameters (realistic for FX)
    omega = 0.00001
    alpha = 0.12  # ARCH effect
    beta = 0.85   # GARCH effect (persistence = 0.97)
    
    returns = np.zeros(n_days)
    sigma2 = np.zeros(n_days)
    sigma2[0] = omega / (1 - alpha - beta)
    
    # Generate GARCH process
    for t in range(1, n_days):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
    
    # Add occasional jumps (fat tails)
    jump_prob = 0.01
    jumps = np.random.binomial(1, jump_prob, n_days)
    jump_sizes = np.random.normal(0, 0.015, n_days)
    returns += jumps * jump_sizes
    
    # Create FX price series
    fx_levels = 1.10 * np.exp(np.cumsum(returns))
    fx_series = pd.Series(fx_levels, index=dates, name='EURUSD')
    
    return fx_series


if __name__ == "__main__":
    
    print("Generating realistic FX data with GARCH volatility clustering...")
    fx_data = generate_realistic_fx_data(n_days=1510, start_date='2020-01-01')
    
    print(f"\nData generated:")
    print(f"  Start: {fx_data.index[0]}")
    print(f"  End: {fx_data.index[-1]}")
    print(f"  Observations: {len(fx_data)}")
    print(f"  Initial level: {fx_data.iloc[0]:.4f}")
    print(f"  Final level: {fx_data.iloc[-1]:.4f}")
    
    # Create validator instance
    validator = FXCalibrationValidator(fx_data, name="EURUSD")
    
    # Run all tests
    summary = validator.run_all_calibration_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print()
    
    # Print detailed results
    validator.print_detailed_results()
    
    # Print specific insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Volatility comparison
    vol_result = validator.results['volatility_scaling']
    print(f"\n1. VOLATILITY SCALING (√5 rule):")
    print(f"   σ from 1-day: {vol_result['sigma_annual_from_1d']*100:.2f}%")
    print(f"   σ from 5-day: {vol_result['sigma_annual_from_5d']*100:.2f}%")
    print(f"   Deviation: {vol_result['deviation_pct']:+.2f}%")
    print(f"   Conclusion: {vol_result['conclusion']}")
    
    # ARCH effects
    arch_result = validator.results['arch_effects']
    print(f"\n2. ARCH EFFECTS (Volatility Clustering):")
    print(f"   LM Statistic: {arch_result['lm_statistic']:.2f}")
    print(f"   P-value: {arch_result['lm_p_value']:.4f}")
    print(f"   Conclusion: {arch_result['conclusion']}")
    
    # GARCH persistence
    if 'error' not in validator.results['garch']:
        garch_result = validator.results['garch']
        print(f"\n3. GARCH PERSISTENCE:")
        print(f"   α (ARCH): {garch_result['alpha']:.4f}")
        print(f"   β (GARCH): {garch_result['beta']:.4f}")
        print(f"   Persistence (α+β): {garch_result['persistence']:.4f}")
        print(f"   Conclusion: {garch_result['conclusion']}")
    
    # Spike smoothing
    spike_result = validator.results['spike_smoothing']
    print(f"\n4. SPIKE SMOOTHING:")
    print(f"   1-day spikes detected: {spike_result['n_spikes_1d']}")
    print(f"   5-day capture rate: {spike_result['spike_capture_rate_5d']:.1f}%")
    print(f"   Smoothing effect: {spike_result['smoothing_effect_pct']:.1f}%")
    print(f"   Conclusion: {spike_result['conclusion']}")
    
    # PFE impact
    pfe_result = validator.results['pfe_impact']
    print(f"\n5. PFE IMPACT ($1B notional):")
    print(f"   PFE from 1-day: ${pfe_result['pfe_from_1d_dollar']/1e6:.2f}M")
    print(f"   PFE from 5-day: ${pfe_result['pfe_from_5d_dollar']/1e6:.2f}M")
    print(f"   Difference: ${pfe_result['pfe_difference_dollar']/1e6:+.2f}M ({pfe_result['pfe_difference_pct']:+.2f}%)")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80 + "\n")