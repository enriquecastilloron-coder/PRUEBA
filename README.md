# Fatigue Analysis - Streamlit Web Application

📊 **Interactive web application for fatigue life analysis** using Weibull distribution with Maximum Likelihood Estimation and Bayesian Inference (PyMC).

## 🌟 Features

- **Data Input**: Upload your own data (CSV, Excel, R format) or use Holmen example
- **MLE Analysis**: Maximum Likelihood parameter estimation
- **Bayesian MCMC**: Full posterior sampling with PyMC
- **Interactive Results**: Trace plots, posterior distributions, and percentile curves
- **Professional UI**: Clean, organized interface with Streamlit

## 🚀 Quick Start

### Option 1: Run Locally

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the app**:
```bash
streamlit run app.py
```

3. **Open browser** at `http://localhost:8501`

### Option 2: Deploy to Streamlit Cloud (FREE & PERMANENT) ⭐

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Fatigue Analysis App"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fatigue-analysis.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Click "Deploy"

3. **Your app will be live** at:
   `https://YOUR_USERNAME-fatigue-analysis.streamlit.app`

### Option 3: Deploy to HuggingFace Spaces (Also FREE)

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (Streamlit)
3. Upload `app.py` and `requirements.txt`
4. Your app goes live automatically!

## 📁 Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🎯 How to Use

1. **Data Tab**: Load data (example or upload file)
2. **Analysis Tab**: Run MLE → Run MCMC
3. **Results Tab**: View posteriors and percentile curves
4. **About Tab**: Read documentation

## 📊 Data Format

### CSV/Excel Format:
```csv
N,Deltasigma
100,0.95
200,0.90
...
```

### R/OpenBUGS Format:
```r
list(
  M=75,
  N=c(100, 200, 300, ...),
  Deltasigma=c(0.95, 0.90, 0.85, ...),
  minlambda=-10,
  maxlambda=-6,
  ...
)
```

## ⚙️ Configuration

Adjust parameters in the sidebar:
- **MCMC Settings**: Warmup, draws, chains, target accept
- **Percentile Settings**: Stress range, points, samples

## 📚 References

- Castillo, E., & Fernández-Canteli, A. (2009). *A Unified Statistical Methodology for Modeling Fatigue Damage*.
- Holmen, J.O. (1979). *Fatigue of Concrete by Constant and Variable Amplitude Loading*.

## 🛠️ Tech Stack

- **Streamlit** - Web framework
- **PyMC** - Bayesian inference
- **ArviZ** - MCMC diagnostics
- **NumPy/Pandas** - Data handling
- **Matplotlib** - Visualization

## 📝 License

MIT License - feel free to use and modify!

## 👤 Author

Created by Enrique Castillo Ron

---

**Need help?** Open an issue or contact via GitHub.
