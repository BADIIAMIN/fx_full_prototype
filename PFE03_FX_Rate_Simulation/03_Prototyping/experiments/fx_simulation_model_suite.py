"""
FX Simulation Model Validation Suite
Purpose: Validate constant-volatility GBM assumptions for PFE/XVA
Author: Model Risk Management
Date: 2026-02-09
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, anderson, kstest, norm, chi2
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# TEST 1: RETURN DISTRIBUTION TESTS
# ============================================================================

class FXDistributionValidator:
    """
    Validates distributional assumptions of FX returns under GBM
    """
    
    def __init__(self, fx_data: pd.Series, name: str = "EURUSD"):
        """
        Parameters:
        -----------
        fx_data : pd.Series
            Time series of FX spot rates (daily frequency expected)
        name : str
            Currency pair name for reporting
        """
        self.fx_data = fx_data
        self.name = name
        self.returns = None
        self.log_returns = None
        self.standardized_residuals = None
        self.results = {}
        
    def compute_returns(self, return_type: str = 'log', frequency: int = 1):
        """
        Compute returns with specified frequency
        
        Parameters:
        -----------
        return_type : str
            'log' for log returns, 'simple' for arithmetic returns
        frequency : int
            Number of days (1 for daily, 5 for weekly, 21 for monthly)
        """
        if return_type == 'log':
            self.log_returns = np.log(self.fx_data / self.fx_data.shift(frequency)).dropna()
        else:
            self.returns = (self.fx_data / self.fx_data.shift(frequency) - 1).dropna()
        
        return self.log_returns if return_type == 'log' else self.returns
    
    def compute_standardized_residuals(self, mu: float = None, sigma: float = None, dt: float = 1/252):
        """
        Compute standardized residuals: z = (r - μ) / σ√dt
        
        Parameters:
        -----------
        mu : float
            Drift parameter (if None, uses sample mean)
        sigma : float
            Volatility parameter (if None, uses sample std)
        dt : float
            Time step in years (1/252 for daily)
        """
        if self.log_returns is None:
            self.compute_returns()
        
        mu = mu if mu is not None else self.log_returns.mean()
        sigma = sigma if sigma is not None else self.log_returns.std()
        
        self.standardized_residuals = (self.log_returns - mu) / (sigma * np.sqrt(dt))
        
        return self.standardized_residuals
    
    def jarque_bera_test(self) -> Dict:
        """
        Test for normality using Jarque-Bera test
        H0: Data is normally distributed
        
        Returns:
        --------
        dict with test statistic, p-value, and interpretation
        """
        if self.standardized_residuals is None:
            self.compute_standardized_residuals()
        
        jb_stat, jb_pval = jarque_bera(self.standardized_residuals)
        
        result = {
            'test_name': 'Jarque-Bera',
            'statistic': jb_stat,
            'p_value': jb_pval,
            'conclusion': 'PASS' if jb_pval > 0.05 else 'FAIL',
            'severity': self._classify_severity(jb_pval, [0.01, 0.05]),
            'interpretation': self._interpret_jb(jb_pval)
        }
        
        self.results['jarque_bera'] = result
        return result
    
    def compute_kurtosis(self, excess: bool = True) -> Dict:
        """
        Compute kurtosis (4th moment)
        
        Parameters:
        -----------
        excess : bool
            If True, returns excess kurtosis (kurtosis - 3)
        
        Returns:
        --------
        dict with kurtosis value and interpretation
        """
        if self.standardized_residuals is None:
            self.compute_standardized_residuals()
        
        kurt_value = stats.kurtosis(self.standardized_residuals, fisher=excess)
        
        # For fisher=True, normal is 0; for False, normal is 3
        threshold_low = -0.5 if excess else 2.5
        threshold_high = 1.0 if excess else 4.0
        threshold_critical = 2.0 if excess else 5.0
        
        result = {
            'test_name': 'Kurtosis',
            'value': kurt_value,
            'excess_kurtosis': kurt_value if excess else kurt_value - 3,
            'conclusion': 'PASS' if threshold_low <= kurt_value <= threshold_high else 'FAIL',
            'severity': self._classify_kurtosis_severity(kurt_value if excess else kurt_value - 3),
            'interpretation': self._interpret_kurtosis(kurt_value if excess else kurt_value - 3)
        }
        
        self.results['kurtosis'] = result
        return result
    
    def anderson_darling_test(self) -> Dict:
        """
        Anderson-Darling test for normality (more sensitive to tails than KS)
        H0: Data comes from normal distribution
        
        Returns:
        --------
        dict with test results and interpretation
        """
        if self.standardized_residuals is None:
            self.compute_standardized_residuals()
        
        ad_result = anderson(self.standardized_residuals, dist='norm')
        
        # Critical values at [15%, 10%, 5%, 2.5%, 1%]
        critical_idx = 2  # 5% significance level
        
        result = {
            'test_name': 'Anderson-Darling',
            'statistic': ad_result.statistic,
            'critical_values': ad_result.critical_values,
            'significance_levels': ad_result.significance_level,
            'critical_value_5pct': ad_result.critical_values[critical_idx],
            'conclusion': 'PASS' if ad_result.statistic < ad_result.critical_values[critical_idx] else 'FAIL',
            'severity': self._classify_ad_severity(ad_result.statistic, ad_result.critical_values),
            'interpretation': self._interpret_ad(ad_result.statistic, ad_result.critical_values[critical_idx])
        }
        
        self.results['anderson_darling'] = result
        return result
    
    def qq_plot(self, save_path: str = None) -> Dict:
        """
        Generate Q-Q plot and compute tail deviations
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot (if None, displays only)
        
        Returns:
        --------
        dict with tail deviation metrics
        """
        if self.standardized_residuals is None:
            self.compute_standardized_residuals()
        
        # Generate Q-Q plot
        fig, ax = plt.subplots(figsize=(10, 10))
        stats.probplot(self.standardized_residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {self.name} Standardized Residuals vs Normal', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Compute tail deviations
        theoretical_quantiles = np.percentile(norm.rvs(size=10000), [1, 5, 95, 99])
        empirical_quantiles = np.percentile(self.standardized_residuals, [1, 5, 95, 99])
        
        deviations = {
            'q01_deviation': abs(empirical_quantiles[0] - theoretical_quantiles[0]),
            'q05_deviation': abs(empirical_quantiles[1] - theoretical_quantiles[1]),
            'q95_deviation': abs(empirical_quantiles[2] - theoretical_quantiles[2]),
            'q99_deviation': abs(empirical_quantiles[3] - theoretical_quantiles[3])
        }
        
        # Acceptable threshold: < 0.3 for 5th/95th, < 0.5 for 1st/99th
        result = {
            'test_name': 'Q-Q Plot',
            'tail_deviations': deviations,
            'q01_pct_deviation': deviations['q01_deviation'] / abs(theoretical_quantiles[0]) * 100,
            'q99_pct_deviation': deviations['q99_deviation'] / abs(theoretical_quantiles[3]) * 100,
            'conclusion': 'PASS' if max(deviations['q01_deviation'], deviations['q99_deviation']) < 0.5 else 'FAIL',
            'severity': self._classify_qq_severity(deviations),
            'interpretation': self._interpret_qq(deviations)
        }
        
        self.results['qq_plot'] = result
        return result
    
    def run_all_distribution_tests(self) -> pd.DataFrame:
        """
        Run all distribution tests and return summary
        """
        self.compute_returns(return_type='log', frequency=1)
        self.compute_standardized_residuals()
        
        self.jarque_bera_test()
        self.compute_kurtosis()
        self.anderson_darling_test()
        self.qq_plot()
        
        # Create summary table
        summary_data = []
        for test_name, test_result in self.results.items():
            summary_data.append({
                'Test': test_result['test_name'],
                'Statistic/Value': test_result.get('statistic', test_result.get('value', 'N/A')),
                'Conclusion': test_result['conclusion'],
                'Severity': test_result['severity'],
                'Interpretation': test_result['interpretation']
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    # Helper methods for interpretation
    def _classify_severity(self, p_value: float, thresholds: List[float]) -> str:
        if p_value >= thresholds[1]:  # > 0.05
            return 'LOW'
        elif p_value >= thresholds[0]:  # 0.01 - 0.05
            return 'MEDIUM'
        else:  # < 0.01
            return 'HIGH'
    
    def _interpret_jb(self, p_value: float) -> str:
        if p_value > 0.05:
            return f"Cannot reject normality (p={p_value:.4f}). Data consistent with Gaussian."
        elif p_value > 0.01:
            return f"Weak evidence against normality (p={p_value:.4f}). Consider monitoring."
        else:
            return f"Strong evidence against normality (p={p_value:.4f}). Model may underestimate tail risk."
    
    def _classify_kurtosis_severity(self, excess_kurt: float) -> str:
        if -0.5 <= excess_kurt <= 1.0:
            return 'LOW'
        elif 1.0 < excess_kurt <= 2.0:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _interpret_kurtosis(self, excess_kurt: float) -> str:
        if excess_kurt < -0.5:
            return f"Platykurtic (excess={excess_kurt:.2f}). Thinner tails than normal."
        elif excess_kurt <= 1.0:
            return f"Near-normal (excess={excess_kurt:.2f}). Acceptable for GBM."
        elif excess_kurt <= 2.0:
            return f"Moderate fat tails (excess={excess_kurt:.2f}). Consider vol overlay."
        else:
            return f"Material fat tails (excess={excess_kurt:.2f}). GBM may underestimate extremes."
    
    def _classify_ad_severity(self, stat: float, critical_values: np.ndarray) -> str:
        if stat < critical_values[2]:  # 5%
            return 'LOW'
        elif stat < critical_values[4]:  # 1%
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _interpret_ad(self, stat: float, crit_5pct: float) -> str:
        if stat < crit_5pct:
            return f"Passes normality (stat={stat:.3f} < crit={crit_5pct:.3f}). Tails consistent with Gaussian."
        else:
            return f"Fails normality (stat={stat:.3f} > crit={crit_5pct:.3f}). Tail deviations detected."
    
    def _classify_qq_severity(self, deviations: Dict) -> str:
        max_tail_dev = max(deviations['q01_deviation'], deviations['q99_deviation'])
        if max_tail_dev < 0.3:
            return 'LOW'
        elif max_tail_dev < 0.5:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _interpret_qq(self, deviations: Dict) -> str:
        max_tail_dev = max(deviations['q01_deviation'], deviations['q99_deviation'])
        if max_tail_dev < 0.3:
            return f"Tail fit excellent (max dev={max_tail_dev:.3f}). Model adequate for PFE."
        elif max_tail_dev < 0.5:
            return f"Tail fit acceptable (max dev={max_tail_dev:.3f}). Monitor during stress."
        else:
            return f"Tail fit poor (max dev={max_tail_dev:.3f}). Consider stochastic vol or overlays."


# ============================================================================
# TEST 2: PIT (PROBABILITY INTEGRAL TRANSFORM) TEST
# ============================================================================

class PITValidator:
    """
    Validates model calibration via Probability Integral Transform
    """
    
    def __init__(self, standardized_residuals: np.ndarray, name: str = "EURUSD"):
        self.z = standardized_residuals
        self.name = name
        self.u = None
        self.results = {}
    
    def compute_pit(self) -> np.ndarray:
        """
        Transform standardized residuals to uniform distribution
        u = Φ(z) should be U(0,1) if model is correct
        """
        self.u = norm.cdf(self.z)
        return self.u
    
    def ks_uniformity_test(self) -> Dict:
        """
        Kolmogorov-Smirnov test for uniformity
        H0: u ~ Uniform(0,1)
        """
        if self.u is None:
            self.compute_pit()
        
        ks_stat, ks_pval = kstest(self.u, 'uniform')
        
        result = {
            'test_name': 'KS Uniformity',
            'statistic': ks_stat,
            'p_value': ks_pval,
            'conclusion': 'PASS' if ks_pval > 0.05 else 'FAIL',
            'severity': 'LOW' if ks_pval > 0.05 else ('MEDIUM' if ks_pval > 0.01 else 'HIGH'),
            'interpretation': self._interpret_ks(ks_pval)
        }
        
        self.results['ks_test'] = result
        return result
    
    def plot_pit_histogram(self, bins: int = 20, save_path: str = None) -> Dict:
        """
        Plot PIT histogram (should be flat for uniform)
        """
        if self.u is None:
            self.compute_pit()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        axes[0].hist(self.u, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Theoretical Uniform(0,1)')
        axes[0].set_xlabel('PIT Values', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title(f'{self.name}: PIT Histogram (Should be Flat)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot for uniformity
        theoretical_uniform = np.linspace(0, 1, len(self.u))
        empirical_sorted = np.sort(self.u)
        
        axes[1].scatter(theoretical_uniform, empirical_sorted, alpha=0.5, s=10)
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Uniform')
        axes[1].set_xlabel('Theoretical Uniform Quantiles', fontsize=12)
        axes[1].set_ylabel('Empirical PIT Quantiles', fontsize=12)
        axes[1].set_title('PIT Q-Q Plot vs Uniform(0,1)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Check for clustering (Chi-square test)
        observed, _ = np.histogram(self.u, bins=bins)
        expected = len(self.u) / bins
        chi2_stat = np.sum((observed - expected)**2 / expected)
        chi2_pval = 1 - chi2.cdf(chi2_stat, df=bins-1)
        
        result = {
            'test_name': 'PIT Histogram',
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_pval,
            'conclusion': 'PASS' if chi2_pval > 0.05 else 'FAIL',
            'severity': 'LOW' if chi2_pval > 0.05 else ('MEDIUM' if chi2_pval > 0.01 else 'HIGH'),
            'interpretation': self._interpret_histogram(chi2_pval)
        }
        
        self.results['pit_histogram'] = result
        return result
    
    def kupiec_test(self, alpha: float = 0.05) -> Dict:
        """
        Kupiec test for quantile exceedances
        Tests if exceedances occur with expected frequency
        
        Parameters:
        -----------
        alpha : float
            Quantile level to test (e.g., 0.05 for 95th percentile)
        """
        if self.u is None:
            self.compute_pit()
        
        # Count exceedances
        n = len(self.u)
        exceedances = np.sum(self.u > (1 - alpha))
        expected_exceedances = n * alpha
        
        # Kupiec LR test statistic
        p_hat = exceedances / n
        if p_hat == 0:
            lr_stat = 0
        else:
            lr_stat = -2 * (
                n * (alpha * np.log(alpha) + (1-alpha) * np.log(1-alpha)) -
                (exceedances * np.log(p_hat) + (n - exceedances) * np.log(1 - p_hat))
            )
        
        # LR test ~ Chi-square(1)
        p_value = 1 - chi2.cdf(lr_stat, df=1)
        
        result = {
            'test_name': f'Kupiec ({int((1-alpha)*100)}th percentile)',
            'alpha': alpha,
            'expected_exceedances': expected_exceedances,
            'observed_exceedances': exceedances,
            'exceedance_rate': p_hat,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'conclusion': 'PASS' if p_value > 0.05 else 'FAIL',
            'severity': self._classify_kupiec_severity(abs(exceedances - expected_exceedances), expected_exceedances),
            'interpretation': self._interpret_kupiec(exceedances, expected_exceedances, p_value)
        }
        
        self.results[f'kupiec_{alpha}'] = result
        return result
    
    def run_all_pit_tests(self) -> pd.DataFrame:
        """
        Run all PIT tests
        """
        self.compute_pit()
        self.ks_uniformity_test()
        self.plot_pit_histogram()
        self.kupiec_test(alpha=0.05)
        self.kupiec_test(alpha=0.01)
        
        summary_data = []
        for test_name, test_result in self.results.items():
            summary_data.append({
                'Test': test_result['test_name'],
                'Statistic': test_result.get('statistic', test_result.get('chi2_statistic', test_result.get('lr_statistic', 'N/A'))),
                'P-Value': test_result.get('p_value', test_result.get('chi2_p_value', 'N/A')),
                'Conclusion': test_result['conclusion'],
                'Severity': test_result['severity']
            })
        
        return pd.DataFrame(summary_data)
    
    def _interpret_ks(self, p_value: float) -> str:
        if p_value > 0.05:
            return f"PIT is uniform (p={p_value:.4f}). Model calibration adequate."
        elif p_value > 0.01:
            return f"Weak non-uniformity (p={p_value:.4f}). Monitor calibration stability."
        else:
            return f"PIT deviates from uniform (p={p_value:.4f}). Model may be mis-calibrated."
    
    def _interpret_histogram(self, chi2_pval: float) -> str:
        if chi2_pval > 0.05:
            return f"Histogram flat (p={chi2_pval:.4f}). No clustering detected."
        else:
            return f"Histogram non-uniform (p={chi2_pval:.4f}). Possible volatility clustering or calibration bias."
    
    def _classify_kupiec_severity(self, deviation: float, expected: float) -> str:
        rel_dev = deviation / expected
        if rel_dev < 0.15:
            return 'LOW'
        elif rel_dev < 0.30:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _interpret_kupiec(self, observed: int, expected: float, p_value: float) -> str:
        if p_value > 0.05:
            return f"Exceedances as expected: {observed} vs {expected:.1f} (p={p_value:.4f})."
        else:
            direction = "too many" if observed > expected else "too few"
            return f"Exceedance failure: {observed} vs {expected:.1f} ({direction}, p={p_value:.4f}). Model may underestimate tail risk." if observed > expected else f"Exceedance failure: {observed} vs {expected:.1f} ({direction}, p={p_value:.4f}). Model may be over-conservative."


# ============================================================================
# TEST 3: TAIL RISK VALIDATION - HISTORICAL EXCEEDANCES
# ============================================================================

class TailRiskValidator:
    """
    Validates tail risk coverage of constant-vol GBM
    """
    
    def __init__(self, fx_data: pd.Series, sigma_calibrated: float, name: str = "EURUSD"):
        self.fx_data = fx_data
        self.sigma = sigma_calibrated
        self.name = name
        self.returns = np.log(fx_data / fx_data.shift(1)).dropna()
        self.results = {}
    
    def historical_exceedance_analysis(self, quantiles: List[float] = [0.95, 0.99], 
                                      rolling_window: int = 250) -> Dict:
        """
        Compare realized vs model quantiles over rolling windows
        
        Parameters:
        -----------
        quantiles : List[float]
            Quantiles to test (e.g., [0.95, 0.99])
        rolling_window : int
            Window size in days (252 = 1 year)
        """
        results_by_quantile = {}
        
        for q in quantiles:
            realized_quantiles = []
            model_quantiles = []
            exceedance_ratios = []
            
            # Rolling window analysis
            for i in range(rolling_window, len(self.returns)):
                window_returns = self.returns.iloc[i-rolling_window:i]
                
                # Realized quantile (absolute returns)
                realized_q = np.percentile(np.abs(window_returns), q * 100)
                realized_quantiles.append(realized_q)
                
                # Model quantile: z = Φ^(-1)(q) * σ * √dt
                dt = 1 / 252
                model_q = norm.ppf((1 + q) / 2) * self.sigma * np.sqrt(dt)  # Two-sided
                model_quantiles.append(model_q)
                
                # Exceedance ratio
                ratio = realized_q / model_q if model_q > 0 else np.nan
                exceedance_ratios.append(ratio)
            
            exceedance_ratios = np.array(exceedance_ratios)
            
            # Statistics
            mean_ratio = np.nanmean(exceedance_ratios)
            median_ratio = np.nanmedian(exceedance_ratios)
            std_ratio = np.nanstd(exceedance_ratios)
            pct_over_12 = np.sum(exceedance_ratios > 1.2) / len(exceedance_ratios) * 100
            
            result = {
                'quantile': q,
                'mean_ratio': mean_ratio,
                'median_ratio': median_ratio,
                'std_ratio': std_ratio,
                'pct_over_1.2': pct_over_12,
                'conclusion': 'PASS' if mean_ratio < 1.2 and pct_over_12 < 20 else 'FAIL',
                'severity': self._classify_exceedance_severity(mean_ratio, pct_over_12),
                'interpretation': self._interpret_exceedance(q, mean_ratio, pct_over_12)
            }
            
            results_by_quantile[f'q{int(q*100)}'] = result
        
        self.results['exceedance_analysis'] = results_by_quantile
        return results_by_quantile
    
    def plot_exceedance_ratios(self, quantile: float = 0.99, rolling_window: int = 250, 
                              save_path: str = None):
        """
        Plot time series of exceedance ratios
        """
        realized_quantiles = []
        model_quantiles = []
        dates = []
        
        for i in range(rolling_window, len(self.returns)):
            window_returns = self.returns.iloc[i-rolling_window:i]
            realized_q = np.percentile(np.abs(window_returns), quantile * 100)
            
            dt = 1 / 252
            model_q = norm.ppf((1 + quantile) / 2) * self.sigma * np.sqrt(dt)
            
            realized_quantiles.append(realized_q)
            model_quantiles.append(model_q)
            dates.append(self.returns.index[i])
        
        ratios = np.array(realized_quantiles) / np.array(model_quantiles)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, ratios, linewidth=1.5, label='Realized / Model Ratio', color='steelblue')
        ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect Calibration')
        ax.axhline(1.2, color='orange', linestyle='--', linewidth=2, label='Warning Threshold (1.2)')
        ax.axhline(1.5, color='red', linestyle='--', linewidth=2, label='Critical Threshold (1.5)')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{int(quantile*100)}th Percentile Ratio', fontsize=12)
        ax.set_title(f'{self.name}: Rolling {rolling_window}d Exceedance Ratio (Q{int(quantile*100)})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _classify_exceedance_severity(self, mean_ratio: float, pct_over_12: float) -> str:
        if mean_ratio < 1.15 and pct_over_12 < 10:
            return 'LOW'
        elif mean_ratio < 1.3 and pct_over_12 < 25:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _interpret_exceedance(self, quantile: float, mean_ratio: float, pct_over_12: float) -> str:
        if mean_ratio < 1.15:
            return f"Q{int(quantile*100)}: Model adequately covers tails (mean ratio={mean_ratio:.2f})."
        elif mean_ratio < 1.3:
            return f"Q{int(quantile*100)}: Model underestimates tails moderately (mean ratio={mean_ratio:.2f}). Consider vol overlay."
        else:
            return f"Q{int(quantile*100)}: Material tail underestimation (mean ratio={mean_ratio:.2f}, {pct_over_12:.1f}% exceed 1.2). Requires model enhancement."


# ============================================================================
# TEST 4: STRESS PERIOD ANALYSIS
# ============================================================================

class StressValidator:
    """
    Validate model performance during known stress periods
    """
    
    def __init__(self, fx_data: pd.Series, sigma_calibrated: float, name: str = "EURUSD"):
        self.fx_data = fx_data
        self.sigma = sigma_calibrated
        self.name = name
        self.returns = (fx_data / fx_data.shift(1) - 1).dropna()
        
        # Define stress periods
        self.stress_periods = {
            'COVID': ('2020-02-15', '2020-04-15'),
            'Brexit': ('2016-06-15', '2016-07-15'),
            'Ukraine': ('2022-02-15', '2022-03-31')
        }
    
    def stress_coverage_test(self, horizon_days: int = 30, quantile: float = 0.99) -> Dict:
        """
        Test if model PFE covers realized max moves during stress
        
        Parameters:
        -----------
        horizon_days : int
            Horizon for stress test (e.g., 30 days)
        quantile : float
            Quantile level (e.g., 0.99 for PFE₉₉)
        """
        results_by_period = {}
        
        for period_name, (start, end) in self.stress_periods.items():
            try:
                # Extract stress period returns
                stress_returns = self.returns.loc[start:end]
                
                if len(stress_returns) == 0:
                    results_by_period[period_name] = {
                        'status': 'NO_DATA',
                        'interpretation': f"No data available for {period_name} period"
                    }
                    continue
                
                # Realized maximum absolute move
                realized_max = stress_returns.abs().max()
                
                # Model quantile for horizon
                dt = horizon_days / 252
                model_move = norm.ppf(quantile) * self.sigma * np.sqrt(dt)
                
                # Coverage test
                coverage = (model_move >= realized_max)
                coverage_ratio = model_move / realized_max if realized_max > 0 else np.inf
                
                result = {
                    'period': period_name,
                    'start_date': start,
                    'end_date': end,
                    'realized_max_move': realized_max,
                    'model_pfe99_move': model_move,
                    'coverage_ratio': coverage_ratio,
                    'coverage': coverage,
                    'conclusion': 'PASS' if coverage else 'FAIL',
                    'severity': 'LOW' if coverage_ratio > 1.1 else ('MEDIUM' if coverage_ratio > 0.9 else 'HIGH'),
                    'interpretation': self._interpret_stress_coverage(period_name, coverage_ratio, coverage)
                }
                
                results_by_period[period_name] = result
                
            except Exception as e:
                results_by_period[period_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'interpretation': f"Error processing {period_name}: {str(e)}"
                }
        
        self.results['stress_coverage'] = results_by_period
        return results_by_period
    
    def plot_stress_periods(self, save_path: str = None):
        """
        Visualize FX behavior during stress periods
        """
        fig, axes = plt.subplots(len(self.stress_periods), 1, figsize=(14, 4*len(self.stress_periods)))
        
        if len(self.stress_periods) == 1:
            axes = [axes]
        
        for idx, (period_name, (start, end)) in enumerate(self.stress_periods.items()):
            try:
                # Extract period data
                period_fx = self.fx_data.loc[start:end]
                
                axes[idx].plot(period_fx.index, period_fx.values, linewidth=2, color='darkred')
                axes[idx].set_title(f'{self.name} - {period_name} Crisis ({start} to {end})', 
                                  fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('FX Rate', fontsize=10)
                axes[idx].grid(True, alpha=0.3)
                
            except:
                axes[idx].text(0.5, 0.5, f'No data for {period_name}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _interpret_stress_coverage(self, period: str, ratio: float, coverage: bool) -> str:
        if coverage and ratio > 1.2:
            return f"{period}: Model conservative (covers {ratio:.2f}x realized max). Adequate for stress."
        elif coverage:
            return f"{period}: Model marginally covers (ratio={ratio:.2f}). Monitor during similar stress."
        else:
            return f"{period}: Model FAILS to cover (ratio={ratio:.2f}). Underestimates extreme stress risk."


# ============================================================================
# MASTER VALIDATION REPORT GENERATOR
# ============================================================================

class FXModelValidationReport:
    """
    Generates comprehensive validation report
    """
    
    def __init__(self, fx_data: pd.Series, sigma_calibrated: float, name: str = "EURUSD"):
        self.fx_data = fx_data
        self.sigma = sigma_calibrated
        self.name = name
        
        # Initialize validators
        self.dist_validator = FXDistributionValidator(fx_data, name)
        self.tail_validator = TailRiskValidator(fx_data, sigma_calibrated, name)
        self.stress_validator = StressValidator(fx_data, sigma_calibrated, name)
    
    def run_full_validation(self) -> Dict:
        """
        Run all validation tests and generate report
        """
        print(f"\n{'='*80}")
        print(f"FX MODEL VALIDATION REPORT: {self.name}")
        print(f"{'='*80}\n")
        
        # Test 1: Distribution Tests
        print("Running Test 1: Distribution Tests...")
        dist_summary = self.dist_validator.run_all_distribution_tests()
        
        # Test 2: PIT Tests
        print("Running Test 2: PIT Tests...")
        z = self.dist_validator.standardized_residuals
        pit_validator = PITValidator(z, self.name)
        pit_summary = pit_validator.run_all_pit_tests()
        
        # Test 3: Tail Risk
        print("Running Test 3: Tail Risk Analysis...")
        tail_results = self.tail_validator.historical_exceedance_analysis()
        self.tail_validator.plot_exceedance_ratios()
        
        # Test 4: Stress Tests
        print("Running Test 4: Stress Period Analysis...")
        stress_results = self.stress_validator.stress_coverage_test()
        self.stress_validator.plot_stress_periods()
        
        print(f"\n{'='*80}")
        print("VALIDATION COMPLETE")
        print(f"{'='*80}\n")
        
        return {
            'distribution_tests': dist_summary,
            'pit_tests': pit_summary,
            'tail_risk': tail_results,
            'stress_coverage': stress_results
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Generate sample FX data (replace with actual data)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='B')
    
    # Simulate GBM with fat tails (for realistic testing)
    sigma_true = 0.10  # 10% annual vol
    dt = 1/252
    
    # Add some fat tail behavior
    normal_shocks = np.random.normal(0, 1, len(dates)-1)
    jump_shocks = np.random.standard_t(df=5, size=len(dates)-1) * 0.3  # Fat tails
    shocks = 0.7 * normal_shocks + 0.3 * jump_shocks  # Mixture
    
    returns = sigma_true * np.sqrt(dt) * shocks
    fx_levels = 1.10 * np.exp(np.cumsum(returns))
    fx_levels = np.insert(fx_levels, 0, 1.10)
    
    fx_series = pd.Series(fx_levels, index=dates, name='EURUSD')
    
    # Run validation
    sigma_calibrated = 0.10  # From 3-year history
    
    validator = FXModelValidationReport(fx_series, sigma_calibrated, name="EURUSD")
    results = validator.run_full_validation()
    
    # Display summaries
    print("\n" + "="*80)
    print("DISTRIBUTION TEST SUMMARY")
    print("="*80)
    print(results['distribution_tests'].to_string(index=False))
    
    print("\n" + "="*80)
    print("PIT TEST SUMMARY")
    print("="*80)
    print(results['pit_tests'].to_string(index=False))
    
    print("\n" + "="*80)
    print("TAIL RISK SUMMARY")
    print("="*80)
    for quantile, result in results['tail_risk'].items():
        print(f"\n{quantile.upper()}:")
        print(f"  Mean Ratio: {result['mean_ratio']:.3f}")
        print(f"  % Over 1.2: {result['pct_over_1.2']:.1f}%")
        print(f"  Conclusion: {result['conclusion']} ({result['severity']})")
        print(f"  {result['interpretation']}")