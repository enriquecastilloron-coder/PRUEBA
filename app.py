"""
============================================================================
FATIGUE ANALYSIS - STREAMLIT WEB APPLICATION
Weibull Model with MLE + Bayesian Inference
Castillo-Canteli Dimensionless Formulation
============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import warnings
import io
import re
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fatigue Analysis - Weibull Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS - DATA PARSING
# ============================================================================

def parse_r_format(content):
    """Parse R/OpenBUGS format data file"""
    try:
        content = content.strip()
        if content.startswith('list('):
            content = content[5:]
        if content.endswith(')'):
            content = content[:-1]
        
        params = {}
        
        # Extract vectors with c(...)
        vector_pattern = r'(\w+)\s*=\s*c\(([\d\.,\s-]+)\)'
        for match in re.finditer(vector_pattern, content):
            param_name = match.group(1)
            values_str = match.group(2)
            values = [float(x.strip()) for x in values_str.split(',')]
            params[param_name] = values
        
        # Extract scalar parameters
        scalar_pattern = r'(\w+)\s*=\s*([\d\.-]+)(?=[,\s]|$)'
        for match in re.finditer(scalar_pattern, content):
            param_name = match.group(1)
            if param_name not in params:  # Don't overwrite vectors
                value = float(match.group(2))
                params[param_name] = value
        
        return params
    except Exception as e:
        st.error(f"Error parsing R format: {str(e)}")
        return None

def load_data_from_file(uploaded_file):
    """Load data from uploaded file (CSV, Excel, or R format)"""
    try:
        filename = uploaded_file.name.lower()
        
        # Read file content
        if filename.endswith(('.csv', '.xlsx', '.xls')):
            if filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Check required columns
            if 'N' not in df.columns or 'Deltasigma' not in df.columns:
                st.error("CSV/Excel file must contain 'N' and 'Deltasigma' columns")
                return None
            
            data = {
                'N': df['N'].values,
                'Deltasigma': df['Deltasigma'].values,
                'M': len(df)
            }
            
            # Extract other parameters if present
            for col in df.columns:
                if col not in ['N', 'Deltasigma']:
                    data[col] = df[col].values[0] if len(df[col].unique()) == 1 else df[col].values
            
            return data
            
        elif filename.endswith(('.txt', '.dat', '.r')):
            # R/OpenBUGS format
            content = uploaded_file.read().decode('utf-8')
            return parse_r_format(content)
        else:
            st.error("Unsupported file format. Use CSV, Excel, TXT, or DAT files.")
            return None
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# ============================================================================
# HOLMEN EXAMPLE DATA
# ============================================================================

def get_holmen_data():
    """Return Holmen example dataset"""
    return {
        'M': 75,
        'ns': 50,
        'np': 5,
        'minlambda': -10,
        'maxlambda': -6,
        'minDeltasigma0': 0.338,
        'maxDeltasigma0': 0.641,
        'minbeta': 0.9,
        'maxbeta': 10,
        'mindelta': 1,
        'maxdelta': 3.53,
        'minN0': 200,
        'maxN0': 29600000,
        'percentiles': [0.01, 0.10, 0.5, 0.90, 0.99],
        'Deltasigma': [0.950]*15 + [0.900]*15 + [0.825]*15 + [0.750]*15 + [0.675]*15,
        'N': [37,72,74,76,83,85,105,109,120,123,143,203,206,217,257,
              201,216,226,252,257,295,311,342,356,451,457,509,540,680,1129,
              1246,1258,1460,1492,2400,2410,2590,2903,3330,3590,3847,4110,4820,5560,5598,
              6710,9930,12600,15580,16190,17280,18620,20300,24900,26260,27940,36350,48420,50090,67340,
              102950,280320,339830,366900,485620,658960,896330,1241760,1250200,1329780,1399830,1459140,3294820,12709000,14373000]
    }

# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

def weibull_log_likelihood(params, stress, cycles):
    """
    Weibull log-likelihood for fatigue data.
    Castillo-Canteli dimensionless formulation.
    EXACTAMENTE como en el notebook original.
    
    Model for minima (lower bounded data):
    log(N) ~ Weibull with location-scale depending on stress
    
    Parameters:
    -----------
    params : array [N0, Delta0, beta, lambda_param, delta]
        N0: reference number of cycles
        Delta0: reference stress (endurance limit)
        beta: shape parameter (Weibull)
        lambda_param: location parameter
        delta: scale parameter
    """
    N0, Delta0, beta, lambda_param, delta = params
    
    # Validations
    if N0 <= 0 or Delta0 <= 0 or beta <= 0 or delta <= 0:
        return -np.inf
    
    # Dimensionless transformation
    log_N_dimensionless = np.log(cycles) - np.log(N0)
    r = np.log(stress) - np.log(Delta0)
    
    # Check for valid r values
    if np.any(np.abs(r) < 1e-10):
        return -np.inf
    
    # Weibull parameters for minima
    # Location parameter
    mu_Y = (-lambda_param - delta) / r
    # Scale parameter
    sigma_Y = delta / (beta * np.abs(r))
    
    # Check valid sigma
    if np.any(sigma_Y <= 0):
        return -np.inf
    
    # Standardized variable for Weibull (Gumbel for minima parametrization)
    z = (log_N_dimensionless - mu_Y) / sigma_Y
    
    # Weibull log-likelihood for minima
    # CDF: F(y) = 1 - exp(-exp(z)) where z = (y - mu)/sigma
    # PDF: f(y) = (1/sigma) * exp(z - exp(z))
    
    log_lik = -np.log(sigma_Y) + z - np.exp(z)
    
    # Check for invalid values
    if not np.all(np.isfinite(log_lik)):
        return -np.inf
    
    return np.sum(log_lik)

def negative_log_likelihood_func(params, stress, cycles):
    """Negative log-likelihood for minimization."""
    return -weibull_log_likelihood(params, stress, cycles)

def mle_estimation(N, Deltasigma, config):
    """Maximum Likelihood Estimation - EXACTAMENTE como notebook original"""
    
    N_min = np.min(N)
    stress_min = np.min(Deltasigma)
    
    # Initial guess based on physical reasoning
    N0_init = N_min * 0.5
    Delta0_init = stress_min * 0.7
    beta_init = 3.0
    lambda_init = -8.0
    delta_init = 2.0
    
    initial_params = np.array([N0_init, Delta0_init, beta_init, lambda_init, delta_init])
    
    # Parameter bounds for optimization
    bounds = [
        (0.001, N_min * 0.9),                               # N0
        (stress_min * 0.4, stress_min * 0.99),              # Delta0
        (0.5, 15.0),                                        # beta
        (-12.0, -4.0),                                      # lambda
        (0.5, 5.0)                                          # delta
    ]
    
    # Run global optimization
    result = differential_evolution(
        negative_log_likelihood_func,
        bounds=bounds,
        args=(Deltasigma, N),
        seed=42,
        maxiter=1000,
        popsize=30,
        tol=1e-7,
        atol=1e-7,
        workers=1,
        updating='deferred',
        polish=True
    )
    
    mle_params = result.x
    
    return {
        'N0': mle_params[0],
        'Deltasigma0': mle_params[1],
        'beta': mle_params[2],
        'lambda': mle_params[3],
        'delta': mle_params[4],
        'success': result.success,
        'nll': result.fun,
        'log_likelihood': -result.fun
    }

def bayesian_inference(N, Deltasigma, config, mle_results):
    """Bayesian inference with PyMC - Weibull distribution with 5 parameters"""
    
    M = len(N)
    N_min = np.min(N)
    stress_min = np.min(Deltasigma)
    
    with pm.Model() as model:
        # Priors - 5 parameters
        N0_prior = pm.Uniform('N0', 
                             lower=config.get('minN0', 0.001), 
                             upper=N_min)
        
        Deltasigma0_prior = pm.Uniform('Deltasigma0',
                                       lower=config['minDeltasigma0'],
                                       upper=config['maxDeltasigma0'])
        
        beta_prior = pm.Uniform('beta',
                               lower=config['minbeta'],
                               upper=config['maxbeta'])
        
        lambda_prior = pm.Uniform('lambda_param',
                                 lower=config['minlambda'],
                                 upper=config['maxlambda'])
        
        delta_prior = pm.Uniform('delta',
                                lower=config['mindelta'],
                                upper=config['maxdelta'])
        
        # Transform to dimensionless log-space
        log_N_dimensionless = pt.log(N) - pt.log(N0_prior)
        r = pt.log(Deltasigma) - pt.log(Deltasigma0_prior)
        
        # Weibull parameters for minima
        mu_Y = (-lambda_prior - delta_prior) / r
        sigma_Y = delta_prior / (beta_prior * pm.math.abs(r) + 1e-8)
        
        # Standardized variable
        z = (log_N_dimensionless - mu_Y) / (sigma_Y + 1e-8)
        
        # Log-likelihood (Weibull for minima, Gumbel parametrization)
        log_lik = -pt.log(sigma_Y + 1e-8) + z - pt.exp(z)
        
        # Total likelihood
        likelihood = pm.Potential('likelihood', pt.sum(log_lik))
        
        # Sample
        trace = pm.sample(
            draws=config.get('mcmc_draws', 2000),
            tune=config.get('mcmc_warmup', 1000),
            chains=config.get('mcmc_chains', 2),
            target_accept=config.get('target_accept', 0.95),
            return_inferencedata=True,
            random_seed=42,
            progressbar=True
        )
    
    return trace, model

def compute_percentile(stress, N0, Delta0, beta, lambda_p, delta, prob):
    """
    Compute N_p for given stress and failure probability (Weibull).
    EXACTAMENTE como en el notebook original.
    """
    if stress <= 0 or N0 <= 0 or Delta0 <= 0 or beta <= 0 or delta <= 0:
        return np.nan
    if prob <= 0 or prob >= 1:
        return np.nan
    
    try:
        r = np.log(stress / Delta0)
        if abs(r) < 1e-10:
            return np.nan
        
        mu_Y = (-lambda_p - delta) / r
        sigma_Y = delta / (beta * abs(r))
        
        # Weibull quantile for minima (Gumbel parametrization)
        z_p = np.log(-np.log(1 - prob))
        Y_p = mu_Y + sigma_Y * z_p
        N_p = N0 * np.exp(Y_p)
        
        if not np.isfinite(N_p) or N_p <= 0:
            return np.nan
        
        if N_p < 1e-6 or N_p > 1e10:
            return np.nan
        
        return N_p
    except:
        return np.nan

def compute_percentiles(trace, config, n_samples=1000):
    """
    Compute percentiles of percentiles
    EXACTAMENTE como en el notebook original
    """
    
    # Extract posterior samples
    N0_samples = trace.posterior['N0'].values.flatten()
    Delta0_samples = trace.posterior['Deltasigma0'].values.flatten()
    beta_samples = trace.posterior['beta'].values.flatten()
    lambda_samples = trace.posterior['lambda_param'].values.flatten()
    delta_samples = trace.posterior['delta'].values.flatten()
    
    total_samples = len(N0_samples)
    
    # Configuration - VALORES EXACTOS DEL NOTEBOOK
    n_stress_points = config.get('stress_points', 50)
    stress_min = config.get('stress_min', 0.3)
    stress_max = config.get('stress_max', 1.0)
    
    # PERCENTILES EXACTOS DEL NOTEBOOK
    percentiles_base = [0.01, 0.15, 0.50, 0.85, 0.99]
    percentiles_sub = [0.01, 0.15, 0.50, 0.85, 0.99]
    
    # Stress range
    stress_range = np.linspace(stress_min, stress_max, n_stress_points)
    
    # Sample indices
    sample_indices = np.random.choice(total_samples, size=min(n_samples, total_samples), replace=False)
    
    # Storage for percentiles of percentiles
    percentiles_of_percentiles = {}
    
    # Loop over base percentiles
    for perc_base_idx, perc_base in enumerate(percentiles_base):
        percentile_matrix = np.zeros((n_stress_points, len(sample_indices)))
        
        # Loop over stress points
        for i, stress in enumerate(stress_range):
            # Loop over parameter samples
            for j, idx in enumerate(sample_indices):
                N0_s = N0_samples[idx]
                Delta0_s = Delta0_samples[idx]
                beta_s = beta_samples[idx]
                lambda_s = lambda_samples[idx]
                delta_s = delta_samples[idx]
                
                N_p = compute_percentile(stress, N0_s, Delta0_s, beta_s, lambda_s, delta_s, perc_base)
                
                if not np.isnan(N_p):
                    percentile_matrix[i, j] = N_p
                else:
                    percentile_matrix[i, j] = np.nan
        
        # Sort each row
        percentile_matrix_sorted = np.sort(percentile_matrix, axis=1)
        
        # Extract sub-percentiles
        percentile_indices = [int(p * (len(sample_indices) - 1)) for p in percentiles_sub]
        perc_of_perc_curves = percentile_matrix_sorted[:, percentile_indices]
        
        percentiles_of_percentiles[perc_base] = perc_of_perc_curves
    
    return stress_range, percentiles_of_percentiles, percentiles_base, percentiles_sub

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Fatigue Analysis - Weibull Model</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Maximum Likelihood Estimation + Bayesian Inference**")
    st.markdown("*Castillo-Canteli Dimensionless Formulation*")
    st.markdown("---")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'mle_done' not in st.session_state:
        st.session_state.mle_done = False
    if 'mcmc_done' not in st.session_state:
        st.session_state.mcmc_done = False
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Holmen Example", "Upload File"],
            key="data_source"
        )
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload data file",
                type=['csv', 'xlsx', 'xls', 'txt', 'dat', 'r'],
                help="CSV/Excel with N and Deltasigma columns, or R/OpenBUGS format"
            )
        
        st.markdown("---")
        
        st.subheader("MCMC Settings")
        mcmc_warmup = st.slider("Warmup samples", 500, 2000, 1000, 100)
        mcmc_draws = st.slider("Draw samples", 1000, 4000, 2000, 500)
        mcmc_chains = st.slider("Chains", 1, 4, 2)
        target_accept = st.slider("Target accept", 0.8, 0.99, 0.95, 0.01)
        
        st.markdown("---")
        
        st.subheader("Percentile Settings")
        
        # Calcular límites automáticamente si hay datos cargados
        if st.session_state.data_loaded:
            data_temp = st.session_state.data
            auto_stress_min = max(min(data_temp['Deltasigma']) - 0.03, 0.01)
            auto_stress_max = max(data_temp['Deltasigma']) + 0.03
            
            # Ajustar rangos del input según los datos
            min_range = min(0.01, auto_stress_min * 0.5)
            max_range = max(2.0, auto_stress_max * 1.5)
            
            stress_min = st.number_input("Min stress", min_range, max_range, auto_stress_min, 0.01,
                                        help="Auto-calculated from data. You can adjust.")
            stress_max = st.number_input("Max stress", min_range, max_range, auto_stress_max, 0.01,
                                        help="Auto-calculated from data. You can adjust.")
        else:
            stress_min = st.number_input("Min stress", 0.01, 2.0, 0.3, 0.05)
            stress_max = st.number_input("Max stress", 0.01, 2.0, 1.0, 0.05)
            
        stress_points = st.slider("Stress points", 20, 200, 50, 10)  # DEFAULT 50
        param_samples = st.slider("Parameter samples", 500, 2000, 1000, 100)
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Data", "🔬 Analysis", "📈 Results", "🧬 Synthetic Data", "ℹ️ About"])
    
    # TAB 1: DATA
    with tab1:
        st.header("Data Input")
        
        if data_source == "Holmen Example":
            if st.button("Load Holmen Example Data", type="primary"):
                st.session_state.data = get_holmen_data()
                st.session_state.data_loaded = True
                st.success("✓ Holmen example data loaded!")
        else:
            if 'uploaded_file' in locals() and uploaded_file is not None:
                if st.button("Process Uploaded File", type="primary"):
                    data = load_data_from_file(uploaded_file)
                    if data is not None:
                        st.session_state.data = data
                        st.session_state.data_loaded = True
                        st.success(f"✓ File '{uploaded_file.name}' loaded successfully!")
        
        if st.session_state.data_loaded:
            st.subheader("Data Summary")
            data = st.session_state.data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of observations", len(data['N']))
            with col2:
                st.metric("Stress range", 
                         f"{min(data['Deltasigma']):.3f} - {max(data['Deltasigma']):.3f}")
            with col3:
                st.metric("Cycles range",
                         f"{min(data['N']):.0f} - {max(data['N']):.0f}")
            
            # Data table
            st.subheader("Data Preview")
            df_preview = pd.DataFrame({
                'N (Cycles)': data['N'][:20],
                'Deltasigma (Stress)': data['Deltasigma'][:20]
            })
            st.dataframe(df_preview, use_container_width=True)
            
            # Data Visualization - Two plots
            st.subheader("Data Visualization")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: S-N Diagram (log scale)
            ax = axes[0]
            unique_stresses = np.unique(data['Deltasigma'])
            colors_plot = plt.cm.rainbow(np.linspace(0, 1, len(unique_stresses)))
            
            for i, stress_level in enumerate(unique_stresses):
                mask = np.array(data['Deltasigma']) == stress_level
                cycles_at_stress = np.array(data['N'])[mask]
                ax.scatter(cycles_at_stress, [stress_level]*np.sum(mask),
                          alpha=0.7, s=50, label=f'{stress_level:.3f}',
                          color=colors_plot[i], edgecolors='black', linewidths=0.5)
            
            ax.set_xlabel('Cycles to Failure (N)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Stress (Δσ)', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.set_title('S-N Data - Weibull Model', fontsize=13, fontweight='bold')
            if len(unique_stresses) <= 10:
                ax.legend(fontsize=9, loc='upper right', title='Stress levels')
            ax.grid(True, alpha=0.3, which='both')
            
            # Plot 2: Log-Log plot
            ax = axes[1]
            ax.scatter(np.log(data['N']), np.log(data['Deltasigma']), 
                      alpha=0.6, s=40, color='darkblue', edgecolors='black', linewidths=0.5)
            ax.set_xlabel('ln(Cycles)', fontsize=12, fontweight='bold')
            ax.set_ylabel('ln(Stress)', fontsize=12, fontweight='bold')
            ax.set_title('Log-Log Plot', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # TAB 2: ANALYSIS
    with tab2:
        st.header("Statistical Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first (Data tab)")
        else:
            data = st.session_state.data
            
            # Prepare configuration
            config = {
                'minlambda': data.get('minlambda', -10),
                'maxlambda': data.get('maxlambda', -6),
                'minDeltasigma0': data.get('minDeltasigma0', 0.3),
                'maxDeltasigma0': data.get('maxDeltasigma0', 1.0),
                'minbeta': data.get('minbeta', 0.5),
                'maxbeta': data.get('maxbeta', 15),
                'mindelta': data.get('mindelta', 0.5),
                'maxdelta': data.get('maxdelta', 5),
                'mcmc_warmup': mcmc_warmup,
                'mcmc_draws': mcmc_draws,
                'mcmc_chains': mcmc_chains,
                'target_accept': target_accept,
                'stress_min': stress_min,
                'stress_max': stress_max,
                'stress_points': stress_points,
                'param_samples': param_samples
            }
            
            # Step 1: MLE
            st.subheader("Step 1: Maximum Likelihood Estimation")
            
            if st.button("Run MLE", type="primary", disabled=st.session_state.mle_done):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text("🔄 Initializing optimization (Differential Evolution)...")
                progress_bar.progress(20)
                
                mle_results = mle_estimation(
                    np.array(data['N']),
                    np.array(data['Deltasigma']),
                    config
                )
                
                progress_text.text("✓ MLE optimization completed!")
                progress_bar.progress(100)
                
                st.session_state.mle_results = mle_results
                st.session_state.mle_done = True
                
                # Clear progress
                progress_text.empty()
                progress_bar.empty()
                
                if mle_results['success']:
                    st.success("✓ MLE completed successfully!")
                    st.info("📊 5 parameters estimated using Maximum Likelihood")
                else:
                    st.warning("⚠️ MLE completed with warnings")
            
            if st.session_state.mle_done:
                mle = st.session_state.mle_results
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**Estimated Parameters (5 total):**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("N₀", f"{mle['N0']:.2f}")
                    st.metric("β (beta)", f"{mle['beta']:.4f}")
                with col2:
                    st.metric("Δσ₀", f"{mle['Deltasigma0']:.4f}")
                    st.metric("δ (delta)", f"{mle['delta']:.4f}")
                with col3:
                    st.metric("λ (lambda)", f"{mle['lambda']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Step 2: Bayesian
            st.subheader("Step 2: Bayesian Inference (MCMC)")
            
            if not st.session_state.mle_done:
                st.info("📌 Complete MLE first")
            else:
                if st.button("Run MCMC", type="primary", disabled=st.session_state.mcmc_done):
                    # Progress indicators
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    status_box = st.empty()
                    
                    progress_text.text(f"🔄 Initializing MCMC sampler...")
                    progress_bar.progress(5)
                    
                    with status_box.container():
                        st.info(f"""
                        **MCMC Configuration:**
                        - Chains: {mcmc_chains}
                        - Warmup samples: {mcmc_warmup} per chain
                        - Draw samples: {mcmc_draws} per chain
                        - Total samples: {mcmc_chains * (mcmc_warmup + mcmc_draws)}
                        - Target accept: {target_accept}
                        """)
                    
                    progress_text.text(f"🔄 Sampling {mcmc_chains} chains...")
                    progress_bar.progress(10)
                    
                    # Run MCMC
                    trace, model = bayesian_inference(
                        np.array(data['N']),
                        np.array(data['Deltasigma']),
                        config,
                        st.session_state.mle_results
                    )
                    
                    progress_text.text("✓ MCMC sampling completed!")
                    progress_bar.progress(100)
                    
                    st.session_state.trace = trace
                    st.session_state.model = model
                    st.session_state.mcmc_done = True
                    
                    # Clear progress indicators
                    progress_text.empty()
                    progress_bar.empty()
                    status_box.empty()
                    
                    st.success("✓ MCMC sampling completed!")
                    
                    # Show sampling statistics
                    total_samples = mcmc_chains * mcmc_draws
                    warmup_total = mcmc_chains * mcmc_warmup
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Posterior samples", f"{total_samples:,}")
                    with col2:
                        st.metric("Warmup samples", f"{warmup_total:,}")
                    with col3:
                        st.metric("Total iterations", f"{mcmc_chains * (mcmc_warmup + mcmc_draws):,}")
                    
                    # AÑADIR: Mostrar medianas posteriores
                    st.markdown("---")
                    st.subheader("📊 Posterior Parameter Estimates")
                    
                    posterior = trace.posterior
                    
                    st.markdown("**Posterior Medians (Bayesian Estimates):**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("N₀ (median)", f"{posterior['N0'].median().values:.6f}")
                        st.metric("β (median)", f"{posterior['beta'].median().values:.3f}")
                    with col2:
                        st.metric("Δσ₀ (median)", f"{posterior['Deltasigma0'].median().values:.6f}")
                        st.metric("δ (median)", f"{posterior['delta'].median().values:.3f}")
                    with col3:
                        st.metric("λ (median)", f"{posterior['lambda_param'].median().values:.3f}")
                    
                    # Diagnósticos de convergencia
                    st.markdown("---")
                    st.subheader("🔍 Convergence Diagnostics")
                    
                    summary = az.summary(trace, hdi_prob=0.95)
                    
                    # Check R-hat
                    rhat_vals = summary['r_hat'].values
                    all_rhat_good = np.all(rhat_vals < 1.01)
                    
                    # Check ESS
                    ess_vals = summary['ess_bulk'].values
                    all_ess_good = np.all(ess_vals > 1000)
                    
                    # Divergences
                    try:
                        divergences = trace.sample_stats['diverging'].values
                        n_divergences = np.sum(divergences)
                    except:
                        n_divergences = 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        status = "✅ Good" if all_rhat_good else "⚠️ Check"
                        st.metric("R-hat < 1.01", status)
                    with col2:
                        status = "✅ Good" if all_ess_good else "⚠️ Low"
                        st.metric("ESS > 1000", status)
                    with col3:
                        status = "✅" if n_divergences == 0 else f"⚠️ {n_divergences}"
                        st.metric("Divergences", status)
                    
                    if n_divergences > 0:
                        div_rate = n_divergences / total_samples * 100
                        st.warning(f"⚠️ Divergence rate: {div_rate:.2f}% - Consider increasing target_accept")
            
            if st.session_state.mcmc_done:
                st.markdown('<div class="success-box">✓ Bayesian inference complete</div>', 
                           unsafe_allow_html=True)
    
    # TAB 3: RESULTS
    with tab3:
        st.header("Results")
        
        if not st.session_state.mcmc_done:
            st.warning("⚠️ Please complete the analysis first (Analysis tab)")
        else:
            trace = st.session_state.trace
            data = st.session_state.data
            
            # Posterior summary
            st.subheader("Posterior Summary")
            summary = az.summary(trace, hdi_prob=0.95)
            st.dataframe(summary, use_container_width=True)
            
            # Trace plots
            st.subheader("MCMC Diagnostics - Trace Plots")
            fig_trace = az.plot_trace(trace, compact=True, figsize=(12, 8))
            
            # Guardar para descarga
            import io
            buf_trace = io.BytesIO()
            fig_trace[0][0].figure.savefig(buf_trace, format='png', dpi=150, bbox_inches='tight')
            buf_trace.seek(0)
            trace_bytes = buf_trace.getvalue()
            
            st.pyplot(fig_trace[0][0].figure)
            plt.close()
            
            st.download_button(
                label="💾 Download Trace Plots (PNG)",
                data=trace_bytes,
                file_name="trace_plots.png",
                mime="image/png",
                key="download_trace"
            )
            
            # Posterior distributions
            st.subheader("Posterior Distributions")
            fig, axes = plt.subplots(3, 2, figsize=(12, 14))
            az.plot_posterior(trace, var_names=['N0'], ax=axes[0, 0])
            az.plot_posterior(trace, var_names=['Deltasigma0'], ax=axes[0, 1])
            az.plot_posterior(trace, var_names=['beta'], ax=axes[1, 0])
            az.plot_posterior(trace, var_names=['lambda_param'], ax=axes[1, 1])
            az.plot_posterior(trace, var_names=['delta'], ax=axes[2, 0])
            axes[2, 1].axis('off')  # Hide unused subplot
            plt.tight_layout()
            
            # Guardar para descarga
            buf_post = io.BytesIO()
            fig.savefig(buf_post, format='png', dpi=150, bbox_inches='tight')
            buf_post.seek(0)
            post_bytes = buf_post.getvalue()
            
            st.pyplot(fig)
            plt.close()
            
            st.download_button(
                label="💾 Download Posterior Distributions (PNG)",
                data=post_bytes,
                file_name="posterior_distributions.png",
                mime="image/png",
                key="download_posterior"
            )
            
            # Percentile curves
            st.subheader("Percentile Curves")
            
            if st.button("Compute Percentile Curves", type="primary"):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                status_box = st.empty()
                
                with status_box.container():
                    st.info(f"""
                    **Percentile Computation:**
                    - Posterior samples: {param_samples}
                    - Stress points: {stress_points}
                    - Stress range: [{stress_min:.2f}, {stress_max:.2f}]
                    - Percentiles: P1, P10, P50, P90, P99
                    """)
                
                progress_text.text(f"🔄 Sampling from posterior ({param_samples} samples)...")
                progress_bar.progress(20)
                
                config = {
                    'stress_min': stress_min,
                    'stress_max': stress_max,
                    'stress_points': stress_points
                }
                
                progress_text.text(f"🔄 Computing percentiles for {stress_points} stress levels...")
                progress_bar.progress(40)
                
                # Compute percentiles - RETORNA 4 VALORES
                stress_levels, percentiles_of_percentiles, percentiles_base, percentiles_sub = compute_percentiles(
                    trace, config, param_samples
                )
                
                progress_text.text("✓ Percentiles computed! Generating plot...")
                progress_bar.progress(80)
                
                # GRÁFICO EXACTO DEL NOTEBOOK
                fig, ax = plt.subplots(figsize=(18, 11), facecolor='white')
                ax.set_facecolor('white')
                
                # Colors from notebook
                colors_base = ['#8B0000', '#FF8C00', '#228B22', '#4169E1', '#8B008B']
                colors_shaded = ['#FFB6B9', '#FFCC80', '#A5D6A7', '#90CAF9', '#CE93D8']
                
                perc_names = [f'P{int(p*100)}' for p in percentiles_base]
                
                # FIRST: Plot shaded regions
                for base_idx, (perc_base, perc_name) in enumerate(zip(percentiles_base, perc_names)):
                    curves = percentiles_of_percentiles[perc_base]
                    
                    curve_p_min = curves[:, 0]  # First sub-percentile
                    curve_p_max = curves[:, -1]  # Last sub-percentile
                    
                    valid_mask = (~np.isnan(curve_p_min)) & (~np.isnan(curve_p_max))
                    
                    if np.sum(valid_mask) > 3:
                        color_shaded = colors_shaded[base_idx % len(colors_shaded)]
                        ax.fill_betweenx(stress_levels[valid_mask],
                                curve_p_min[valid_mask],
                                curve_p_max[valid_mask],
                                color=color_shaded,
                                alpha=0.85,
                                label=f'{perc_name} uncertainty band',
                                zorder=base_idx + 1)
                
                # SECOND: Plot median curves - THICK
                median_idx = len(percentiles_sub) // 2
                for base_idx, (perc_base, perc_name) in enumerate(zip(percentiles_base, perc_names)):
                    curves = percentiles_of_percentiles[perc_base]
                    color_base = colors_base[base_idx % len(colors_base)]
                    
                    curve_median = curves[:, median_idx]
                    
                    valid_mask = ~np.isnan(curve_median)
                    
                    if np.sum(valid_mask) > 3:
                        ax.plot(curve_median[valid_mask], stress_levels[valid_mask],
                               color=color_base, linewidth=3.5,
                               label=f'{perc_name} (median curve)',
                               zorder=50 + base_idx)
                
                # THIRD: Plot observed data on TOP
                ax.scatter(data['N'], data['Deltasigma'], c='black', s=80,
                          alpha=0.9, label='Observed data', zorder=100, marker='o',
                          edgecolors='white', linewidths=1.5)
                
                # Set limits
                cycles_min_plot = min(data['N']) * 0.1
                cycles_max_plot = max(data['N']) * 10.0
                
                ax.set_xlim([cycles_min_plot, cycles_max_plot])
                ax.set_xlabel('Cycles to Failure (N)', fontsize=15, fontweight='bold')
                ax.set_ylabel('Stress (Δσ)', fontsize=15, fontweight='bold')
                ax.set_xscale('log')
                ax.set_title('Percentiles of Percentiles with Uncertainty Bands\nBayesian Weibull Model - Castillo-Canteli Formulation',
                            fontsize=15, fontweight='bold', pad=20)
                
                # Legend
                ax.legend(loc='upper right', fontsize=10, framealpha=0.98, ncol=2,
                         columnspacing=1.0, handlelength=2.5,
                         title='Base Percentiles & Uncertainty Bands',
                         title_fontsize=11)
                
                ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
                ax.set_ylim([min(stress_levels) * 0.95, max(stress_levels) * 1.05])
                
                plt.tight_layout()
                
                # GUARDAR FIGURA EN MEMORIA ANTES de progress
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=250, bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.getvalue()  # Guardar bytes antes de cerrar
                
                progress_bar.progress(100)
                
                # Clear progress
                progress_text.empty()
                progress_bar.empty()
                status_box.empty()
                
                st.pyplot(fig)
                plt.close()
                
                # BOTÓN DE DESCARGA - USA img_bytes guardados
                st.download_button(
                    label="💾 Download Percentiles Figure (PNG)",
                    data=img_bytes,
                    file_name="percentiles_of_percentiles.png",
                    mime="image/png",
                    key="download_percentiles"
                )
                
                # Show computation statistics
                total_evaluations = param_samples * stress_points
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total evaluations", f"{total_evaluations:,}")
                with col2:
                    st.metric("Percentile curves", f"{len(percentiles_base)} base × {len(percentiles_sub)} sub = {len(percentiles_base) * len(percentiles_sub)}")
                
                st.success("✓ Percentile curves generated successfully!")
    
    # TAB 4: SYNTHETIC DATA
    with tab4:
        st.header("🧬 Synthetic Data Generation")
        
        if not st.session_state.mcmc_done:
            st.warning("⚠️ Please complete the Bayesian analysis first (Analysis tab)")
        else:
            st.markdown("""
            Generate synthetic datasets using posterior parameter samples from your Bayesian analysis.
            Each dataset uses a random sample from the posterior distribution.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                n_obs = st.number_input("Observations per dataset", 10, 1000, 75, 10)
            with col2:
                n_datasets = st.number_input("Number of datasets", 1, 100, 5, 1)
            
            if st.button("Generate Synthetic Datasets", type="primary"):
                with st.spinner("Generating synthetic data..."):
                    trace = st.session_state.trace
                    data = st.session_state.data
                    
                    # Extract posterior samples
                    posterior = trace.posterior
                    N0_samples = posterior['N0'].values.flatten()
                    Delta0_samples = posterior['Deltasigma0'].values.flatten()
                    beta_samples = posterior['beta'].values.flatten()
                    lambda_samples = posterior['lambda_param'].values.flatten()
                    delta_samples = posterior['delta'].values.flatten()
                    
                    # Select random posterior samples
                    n_available = len(N0_samples)
                    n_to_generate = min(n_datasets, n_available)
                    selected_indices = np.random.choice(n_available, size=n_to_generate, replace=False)
                    
                    synthetic_datasets = []
                    
                    for idx, sample_idx in enumerate(selected_indices):
                        # Get parameters
                        N0 = N0_samples[sample_idx]
                        Delta0 = Delta0_samples[sample_idx]
                        beta = beta_samples[sample_idx]
                        lambda_param = lambda_samples[sample_idx]
                        delta = delta_samples[sample_idx]
                        
                        # Generate stress levels
                        stress_synth = np.random.uniform(
                            low=min(data['Deltasigma']),
                            high=max(data['Deltasigma']),
                            size=n_obs
                        )
                        
                        # Generate cycles
                        cycles_synth = np.zeros(n_obs)
                        for i in range(n_obs):
                            stress = stress_synth[i]
                            r = np.log(stress / Delta0)
                            mu_Y = (-lambda_param - delta) / r
                            sigma_Y = delta / (beta * abs(r))
                            
                            # Generate random Gumbel
                            u = np.random.uniform(0, 1)
                            z = np.log(-np.log(1 - u))
                            Y = mu_Y + sigma_Y * z
                            N = N0 * np.exp(Y)
                            cycles_synth[i] = N
                        
                        # Create dataframe
                        df_synth = pd.DataFrame({
                            'N': cycles_synth,
                            'Deltasigma': stress_synth
                        }).sort_values('Deltasigma').reset_index(drop=True)
                        
                        synthetic_datasets.append({
                            'data': df_synth,
                            'params': {'N0': N0, 'Delta0': Delta0, 'beta': beta, 
                                      'lambda': lambda_param, 'delta': delta}
                        })
                    
                    st.success(f"✓ Generated {n_to_generate} synthetic datasets!")
                    
                    # Show first dataset as preview
                    st.subheader("Preview: Dataset 1")
                    st.dataframe(synthetic_datasets[0]['data'].head(20))
                    
                    # Create CSV files for download
                    for idx, dataset in enumerate(synthetic_datasets):
                        csv = dataset['data'].to_csv(index=False)
                        st.download_button(
                            label=f"💾 Download Dataset {idx+1} (CSV)",
                            data=csv,
                            file_name=f"synthetic_data_{idx+1:03d}.csv",
                            mime="text/csv",
                            key=f"download_synth_{idx}"
                        )
    
    # TAB 5: ABOUT
    with tab5:
        st.header("About This Application")
        
        st.markdown("""
        ### 📊 Fatigue Analysis with Weibull Model
        
        This application performs fatigue life analysis using:
        
        #### **Model - 5 Parameters**
        - **Castillo-Canteli dimensionless formulation**
        - **Weibull distribution for minima** - Pure Weibull model
        - Physically justified for lower-bounded problems (N > 0)
        - Uses Gumbel parametrization for numerical stability
        
        #### **Model Equations**
        ```
        log_N_dimensionless = log(N) - log(N₀)
        r = log(Δσ) - log(Δσ₀)
        μ_Y = (-λ - δ) / r
        σ_Y = δ / (β|r|)
        z = (log_N_dimensionless - μ_Y) / σ_Y
        z ~ Gumbel(0, 1)  [Weibull for minima]
        ```
        
        #### **Why Weibull?**
        - Fatigue life is inherently a minimum value problem
        - Weibull naturally models the weakest link behavior
        - You control the analysis - no automatic simplifications
        
        #### **Analysis Pipeline**
        1. **Maximum Likelihood Estimation (MLE)**
           - Estimates all 5 parameters simultaneously
           - Uses differential evolution optimization
        
        2. **Bayesian Inference (MCMC)**
           - PyMC probabilistic programming with Weibull
           - NUTS sampler for posterior sampling
           - Quantifies uncertainty in all 5 parameters
        
        3. **Percentile Prediction**
           - Generates S-N curves with uncertainty bands
           - Multiple percentile levels (P1, P10, P50, P90, P99)
        
        #### **5 Parameters**
        - **N₀**: Reference number of cycles
        - **Δσ₀**: Reference stress level
        - **β (beta)**: Shape parameter (controls distribution tail)
        - **λ (lambda)**: Location parameter (log-scale)
        - **δ (delta)**: Dimensionless damage parameter
        
        #### **Data Format**
        - **CSV/Excel**: Columns 'N' (cycles) and 'Deltasigma' (stress)
        - **R/OpenBUGS**: list() format with vectors
        
        #### **References**
        - Castillo, E., & Fernández-Canteli, A. (2009). A Unified Statistical Methodology for Modeling Fatigue Damage.
        - Holmen, J.O. (1979). Fatigue of Concrete by Constant and Variable Amplitude Loading.
        
        ---
        
        **Version 2.0 - 5-Parameter Weibull Model** | Built with Streamlit, PyMC, and ArviZ
        """)

if __name__ == "__main__":
    main()
