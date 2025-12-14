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

def mle_estimation(N, Deltasigma, config):
    """Maximum Likelihood Estimation - Weibull distribution with 5 parameters"""
    
    def negative_log_likelihood(params):
        N0, Deltasigma0, beta, lambda_param, delta = params
        
        if N0 <= 0 or Deltasigma0 <= 0 or beta <= 0 or delta <= 0:
            return 1e10
        
        # Dimensionless variables
        log_N_dimensionless = np.log(N) - np.log(N0)
        r = np.log(Deltasigma) - np.log(Deltasigma0)
        
        # Weibull parameters
        mu_Y = (-lambda_param - delta) / r
        sigma_Y = delta / (beta * np.abs(r) + 1e-8)
        
        if np.any(sigma_Y <= 0):
            return 1e10
        
        # Standardized variable
        z = (log_N_dimensionless - mu_Y) / (sigma_Y + 1e-8)
        
        # Log-likelihood (Weibull for minima, Gumbel parametrization)
        log_lik = -np.log(sigma_Y + 1e-8) + z - np.exp(z)
        
        return -np.sum(log_lik)
    
    # Initial guess
    N_min = np.min(N)
    stress_min = np.min(Deltasigma)
    
    x0 = [
        N_min * 0.5,  # N0
        (config['minDeltasigma0'] + config['maxDeltasigma0']) / 2,  # Deltasigma0
        (config['minbeta'] + config['maxbeta']) / 2,  # beta
        (config['minlambda'] + config['maxlambda']) / 2,  # lambda
        (config['mindelta'] + config['maxdelta']) / 2  # delta
    ]
    
    # Bounds
    bounds = [
        (config.get('minN0', 0.001), N_min),  # N0
        (config['minDeltasigma0'], config['maxDeltasigma0']),  # Deltasigma0
        (config['minbeta'], config['maxbeta']),  # beta
        (config['minlambda'], config['maxlambda']),  # lambda
        (config['mindelta'], config['maxdelta'])  # delta
    ]
    
    # Optimize
    result = differential_evolution(
        negative_log_likelihood,
        bounds,
        seed=42,
        maxiter=1000,
        workers=1
    )
    
    return {
        'N0': result.x[0],
        'Deltasigma0': result.x[1],
        'beta': result.x[2],
        'lambda': result.x[3],
        'delta': result.x[4],
        'success': result.success,
        'nll': result.fun
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

def compute_percentiles(trace, config, n_samples=1000):
    """Compute percentile curves from posterior samples - Weibull with 5 parameters"""
    
    # Extract samples - 5 parameters
    N0_samples = trace.posterior['N0'].values.flatten()
    Deltasigma0_samples = trace.posterior['Deltasigma0'].values.flatten()
    beta_samples = trace.posterior['beta'].values.flatten()
    lambda_samples = trace.posterior['lambda_param'].values.flatten()
    delta_samples = trace.posterior['delta'].values.flatten()
    
    # Sample from posterior
    n_posterior = len(N0_samples)
    indices = np.random.choice(n_posterior, size=n_samples, replace=False)
    
    # Stress levels
    stress_min = config.get('stress_min', 0.3)
    stress_max = config.get('stress_max', 1.0)
    stress_points = config.get('stress_points', 100)
    stress_levels = np.linspace(stress_min, stress_max, stress_points)
    
    # Percentiles to compute
    percentiles = config.get('percentiles', [0.01, 0.10, 0.5, 0.90, 0.99])
    
    results = {p: np.zeros(stress_points) for p in percentiles}
    
    for i, stress in enumerate(stress_levels):
        N_samples = []
        
        for idx in indices:
            N0 = N0_samples[idx]
            Deltasigma0 = Deltasigma0_samples[idx]
            beta = beta_samples[idx]
            lambda_val = lambda_samples[idx]
            delta = delta_samples[idx]
            
            # Dimensionless variables
            r = np.log(stress / Deltasigma0)
            
            # Weibull parameters
            mu_Y = (-lambda_val - delta) / r
            sigma_Y = delta / (beta * np.abs(r) + 1e-8)
            
            # Sample from Gumbel (Weibull for minima in standardized form)
            z_sample = np.random.gumbel(0, 1)  # Standard Gumbel
            log_N_dimensionless = mu_Y + sigma_Y * z_sample
            
            # Transform back to N
            N_sample = N0 * np.exp(log_N_dimensionless)
            N_samples.append(N_sample)
        
        # Compute percentiles
        for p in percentiles:
            results[p][i] = np.percentile(N_samples, p * 100)
    
    return stress_levels, results

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
        stress_min = st.number_input("Min stress", 0.1, 1.0, 0.3, 0.05)
        stress_max = st.number_input("Max stress", 0.1, 1.0, 1.0, 0.05)
        stress_points = st.slider("Stress points", 50, 200, 100, 10)
        param_samples = st.slider("Parameter samples", 500, 2000, 1000, 100)
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data", "🔬 Analysis", "📈 Results", "ℹ️ About"])
    
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
            
            # S-N Plot
            st.subheader("S-N Diagram")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(data['N'], data['Deltasigma'], alpha=0.6, s=50)
            ax.set_xscale('log')
            ax.set_xlabel('Number of Cycles (N)', fontsize=12)
            ax.set_ylabel('Stress Range (Δσ)', fontsize=12)
            ax.set_title('S-N Diagram - Raw Data', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
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
                    
                    progress_text.text(f"🔄 Initializing MCMC ({mcmc_chains} chains, {mcmc_draws} draws)...")
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
                    
                    st.success("✓ MCMC sampling completed!")
                    st.info(f"📊 Sampled {mcmc_chains} chains × {mcmc_draws} draws = {mcmc_chains * mcmc_draws} total samples")
            
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
            fig = az.plot_trace(trace, compact=True, figsize=(12, 8))
            st.pyplot(fig[0][0].figure)
            plt.close()
            
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
            st.pyplot(fig)
            plt.close()
            
            # Percentile curves
            st.subheader("Percentile Curves")
            
            if st.button("Compute Percentile Curves", type="primary"):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text(f"🔄 Computing percentile curves ({param_samples} samples)...")
                progress_bar.progress(20)
                
                config = {
                    'percentiles': [0.01, 0.10, 0.5, 0.90, 0.99],
                    'stress_min': stress_min,
                    'stress_max': stress_max,
                    'stress_points': stress_points
                }
                
                progress_bar.progress(40)
                stress_levels, percentile_results = compute_percentiles(
                    trace, config, param_samples
                )
                
                progress_text.text("✓ Percentiles computed! Generating plot...")
                progress_bar.progress(80)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot data
                ax.scatter(data['N'], data['Deltasigma'], 
                          alpha=0.5, s=30, label='Data', color='gray')
                
                # Plot percentiles
                colors = ['red', 'orange', 'green', 'orange', 'red']
                styles = ['--', '-.', '-', '-.', '--']
                
                for (p, color, style) in zip([0.01, 0.10, 0.5, 0.90, 0.99], 
                                             colors, styles):
                    ax.plot(percentile_results[p], stress_levels,
                           color=color, linestyle=style, linewidth=2,
                           label=f'P{int(p*100)}')
                
                ax.set_xscale('log')
                ax.set_xlabel('Number of Cycles (N)', fontsize=12)
                ax.set_ylabel('Stress Range (Δσ)', fontsize=12)
                ax.set_title('Fatigue Life Percentile Curves', 
                            fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                progress_bar.progress(100)
                progress_text.empty()
                progress_bar.empty()
                
                st.pyplot(fig)
                plt.close()
                
                st.success(f"✓ Percentile curves computed from {param_samples} posterior samples")
    
    # TAB 4: ABOUT
    with tab4:
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
