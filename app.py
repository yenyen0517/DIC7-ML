"""
CRISP-DM Linear Regression Demo  ·  Streamlit app
Single-file, runnable with: streamlit run app.py
"""

import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRISP-DM · Linear Regression",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .phase-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border-left: 5px solid #4fc3f7;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 14px;
    color: #e0f7fa;
    font-size: 1.05rem;
    font-weight: 700;
  }
  .metric-card {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    color: #fff;
  }
  .metric-card .label { font-size: 0.78rem; color: #90caf9; letter-spacing: 1px; }
  .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #4fc3f7; }
  .param-row { background: #1a2744; border-radius: 8px; padding: 10px 14px; margin: 6px 0; }
  .stButton>button {
    background: linear-gradient(135deg, #4fc3f7, #0288d1);
    color: white; border: none; border-radius: 8px;
    padding: 10px 28px; font-weight: 600; font-size: 0.95rem;
    width: 100%; cursor: pointer; transition: opacity .2s;
  }
  .stButton>button:hover { opacity: 0.88; }
  .sidebar-badge {
    background: #1e3a5f; border-radius: 6px;
    padding: 4px 10px; font-size: 0.78rem; color: #4fc3f7;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_data(n: int, noise_var: float, seed: int):
    rng = np.random.default_rng(seed)
    a   = float(rng.uniform(-10, 10))
    b   = float(rng.uniform(-50, 50))
    noise_mean = float(rng.uniform(-10, 10))
    x   = rng.uniform(-100, 100, size=n)
    noise = rng.normal(loc=noise_mean, scale=np.sqrt(noise_var) if noise_var > 0 else 0, size=n)
    y   = a * x + b + noise
    df  = pd.DataFrame({"x": x, "y": y})
    return df, a, b, noise_mean

@st.cache_data(show_spinner=False)
def train_model(n: int, noise_var: float, seed: int, test_size: float):
    df, a_true, b_true, noise_mean = generate_data(n, noise_var, seed)
    X = df[["x"]].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test  = model.predict(X_test_s)

    mse_train  = mean_squared_error(y_train, y_pred_train)
    mse_test   = mean_squared_error(y_test,  y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test  = np.sqrt(mse_test)
    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)

    # Learned parameters in original scale
    a_learned = model.coef_[0] / scaler.scale_[0]
    b_learned = model.intercept_ - model.coef_[0] * scaler.mean_[0] / scaler.scale_[0]

    return {
        "df": df, "a_true": a_true, "b_true": b_true, "noise_mean": noise_mean,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred_train": y_pred_train, "y_pred_test": y_pred_test,
        "mse_train": mse_train, "mse_test": mse_test,
        "rmse_train": rmse_train, "rmse_test": rmse_test,
        "r2_train": r2_train, "r2_test": r2_test,
        "a_learned": a_learned, "b_learned": b_learned,
        "model": model, "scaler": scaler,
    }

def make_scatter_plot(res, show_train=True, show_test=True):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    x_all = np.linspace(res["df"]["x"].min(), res["df"]["x"].max(), 300).reshape(-1, 1)
    x_all_s = res["scaler"].transform(x_all)
    y_line  = res["model"].predict(x_all_s)

    if show_train:
        ax.scatter(res["X_train"], res["y_train"],
                   alpha=0.45, s=18, color="#4fc3f7", label="Train")
    if show_test:
        ax.scatter(res["X_test"], res["y_test"],
                   alpha=0.55, s=22, color="#f48fb1", marker="^", label="Test")

    ax.plot(x_all, y_line, color="#ffeb3b", lw=2.2, label="Regression line")

    ax.set_xlabel("x", color="#ccc")
    ax.set_ylabel("y", color="#ccc")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1a2744", labelcolor="white", fontsize=9)
    fig.tight_layout()
    return fig

def make_residual_plot(res):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    for ax, X, y_true, y_pred, label, color in zip(
        axes,
        [res["X_train"], res["X_test"]],
        [res["y_train"], res["y_test"]],
        [res["y_pred_train"], res["y_pred_test"]],
        ["Train Residuals", "Test Residuals"],
        ["#4fc3f7", "#f48fb1"],
    ):
        ax.set_facecolor("#0e1117")
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=16, color=color)
        ax.axhline(0, color="#ffeb3b", lw=1.5, linestyle="--")
        ax.set_title(label, color="#eee", fontsize=10)
        ax.set_xlabel("Predicted", color="#aaa")
        ax.set_ylabel("Residual", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    n         = st.slider("**Sample size  n**",   100, 1000, 300, 50)
    noise_var = st.slider("**Noise variance**",   0,   1000, 100, 10)
    seed      = st.slider("**Random seed**",      0,   999,  42,  1)
    test_size = st.slider("**Test split  (%)**",  10,  40,   20,  5) / 100
    show_train = st.checkbox("Show train points", True)
    show_test  = st.checkbox("Show test points",  True)

    st.markdown("---")
    generate = st.button("🎲  Generate & Train")

    st.markdown("---")
    st.markdown('<span class="sidebar-badge">CRISP-DM · Linear Regression</span>',
                unsafe_allow_html=True)

# ── Auto-run on first load ─────────────────────────────────────────────────────
if "result" not in st.session_state or generate:
    with st.spinner("Training model…"):
        st.session_state["result"] = train_model(n, noise_var, seed, test_size)

res = st.session_state["result"]

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='text-align:center; background:linear-gradient(135deg,#4fc3f7,#f48fb1);
   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
   font-size:2.1rem; font-weight:800; margin-bottom:4px;'>
   📈 Linear Regression · CRISP-DM Workflow
</h1>
<p style='text-align:center; color:#90a4ae; font-size:0.9rem; margin-top:0;'>
   Interactive demo built with scikit-learn &amp; Streamlit
</p>
""", unsafe_allow_html=True)

# CRISP-DM progress bar
phases = ["1 · Business\nUnderstanding", "2 · Data\nUnderstanding",
          "3 · Data\nPreparation", "4 · Modelling",
          "5 · Evaluation", "6 · Deployment"]
cols_ph = st.columns(6)
for col, ph in zip(cols_ph, phases):
    col.markdown(f"""
    <div style='background:#1e3a5f;border-radius:8px;padding:8px 4px;
    text-align:center;font-size:0.72rem;color:#4fc3f7;font-weight:600;
    line-height:1.4;'>{ph}</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 – Business Understanding
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📋  Phase 1 · Business Understanding", expanded=True):
    st.markdown('<div class="phase-header">🎯 Goal: Predict y from x using simple linear regression</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("""
**Problem statement**
- Fit a linear model  **y = ax + b + ε**  to synthetic data
- Understand the role of noise and sample size on model quality
- Evaluate performance on held-out data

**Success criteria**
- Low RMSE on the test set
- R² close to 1.0 for high signal-to-noise data
""")
    with c2:
        st.markdown(r"""
**Data generation formula**

$$y = ax + b + \varepsilon$$

| Symbol | Domain |
|--------|--------|
| x | Uniform(−100, 100) |
| a | Uniform(−10, 10) |
| b | Uniform(−50, 50) |
| ε | Normal(μ∈[−10,10], σ²∈[0,1000]) |
""")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 – Data Understanding
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔍  Phase 2 · Data Understanding", expanded=True):
    st.markdown('<div class="phase-header">📊 Explore the generated dataset</div>',
                unsafe_allow_html=True)

    df = res["df"]
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Samples", f"{len(df):,}"),
        ("x  range", f"[{df['x'].min():.1f}, {df['x'].max():.1f}]"),
        ("y  mean",  f"{df['y'].mean():.2f}"),
        ("y  std",   f"{df['y'].std():.2f}"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("**Raw scatter  (first 500 points)**")
        fig_raw, ax_raw = plt.subplots(figsize=(7, 3.5))
        fig_raw.patch.set_facecolor("#0e1117")
        ax_raw.set_facecolor("#0e1117")
        sample = df.sample(min(500, len(df)), random_state=seed)
        ax_raw.scatter(sample["x"], sample["y"], s=14, alpha=0.5, color="#4fc3f7")
        ax_raw.set_xlabel("x", color="#ccc"); ax_raw.set_ylabel("y", color="#ccc")
        ax_raw.tick_params(colors="#aaa")
        for sp in ax_raw.spines.values(): sp.set_edgecolor("#333")
        fig_raw.tight_layout()
        st.pyplot(fig_raw, width="stretch")
        plt.close(fig_raw)
    with c2:
        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe().round(3), width="stretch")

    st.markdown("**True parameters (hidden from model)**")
    pc1, pc2, pc3 = st.columns(3)
    for col, lbl, val in zip([pc1, pc2, pc3],
                              ["True slope  a", "True intercept  b", "Noise mean  μ"],
                              [res["a_true"], res["b_true"], res["noise_mean"]]):
        col.markdown(f"""<div class="metric-card">
            <div class="label">{lbl}</div>
            <div class="value">{val:.4f}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 – Data Preparation
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🛠️  Phase 3 · Data Preparation", expanded=True):
    st.markdown('<div class="phase-header">⚙️ Feature engineering & train/test split</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Steps performed**
1. Feature matrix **X** = column vector of `x`
2. Target vector **y** unchanged
3. `train_test_split` with `test_size = {test_size:.0%}` · `random_state = {seed}`
4. `StandardScaler` fitted **only** on train set → applied to both splits

**Why scale?**
- Ensures gradient-based solvers converge stably
- Makes coefficients comparable across features
""")
    with c2:
        n_train = len(res["X_train"])
        n_test  = len(res["X_test"])
        st.markdown(f"""
| Set | Samples | Fraction |
|-----|---------|----------|
| Train | {n_train:,} | {n_train/len(df):.0%} |
| Test  | {n_test:,}  | {n_test/len(df):.0%}  |

**Scaler statistics (train)**

| Stat | Value |
|------|-------|
| Mean | {res["scaler"].mean_[0]:.4f} |
| Std  | {res["scaler"].scale_[0]:.4f} |
""")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 – Modelling
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🤖  Phase 4 · Modelling", expanded=True):
    st.markdown('<div class="phase-header">📐 Fit LinearRegression on scaled train data</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.pyplot(make_scatter_plot(res, show_train, show_test), width="stretch")
    with c2:
        st.markdown("**Learned vs True parameters**")
        param_data = {
            "Parameter": ["Slope (a)", "Intercept (b)"],
            "True":    [f"{res['a_true']:.4f}",    f"{res['b_true']:.4f}"],
            "Learned": [f"{res['a_learned']:.4f}", f"{res['b_learned']:.4f}"],
            "Error":   [f"{abs(res['a_true']-res['a_learned']):.4f}",
                        f"{abs(res['b_true']-res['b_learned']):.4f}"],
        }
        st.dataframe(pd.DataFrame(param_data), width="stretch", hide_index=True)

        st.markdown(fr"""
**Model equation** (original scale)

$$\hat{y} = {res['a_learned']:.4f}\, x + {res['b_learned']:.4f}$$

**sklearn coefficients** (scaled space)

| | Value |
|---|---|
| coef_ | {res['model'].coef_[0]:.6f} |
| intercept_ | {res['model'].intercept_:.6f} |
""")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 – Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("📊  Phase 5 · Evaluation", expanded=True):
    st.markdown('<div class="phase-header">✅ Model performance metrics</div>',
                unsafe_allow_html=True)

    metric_cols = st.columns(6)
    evals = [
        ("MSE · Train",  f"{res['mse_train']:.2f}"),
        ("MSE · Test",   f"{res['mse_test']:.2f}"),
        ("RMSE · Train", f"{res['rmse_train']:.2f}"),
        ("RMSE · Test",  f"{res['rmse_test']:.2f}"),
        ("R² · Train",   f"{res['r2_train']:.4f}"),
        ("R² · Test",    f"{res['r2_test']:.4f}"),
    ]
    for col, (lbl, val) in zip(metric_cols, evals):
        col.markdown(f"""<div class="metric-card">
            <div class="label">{lbl}</div>
            <div class="value">{val}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.pyplot(make_residual_plot(res), width="stretch")
    plt.close("all")

    # Interpretation
    r2 = res["r2_test"]
    quality = "Excellent 🎉" if r2 > 0.9 else ("Good 👍" if r2 > 0.7 else
              ("Moderate ⚠️" if r2 > 0.5 else "Poor ❌"))
    st.info(f"**Test R² = {r2:.4f}** → {quality}  |  "
            f"Noise variance = {noise_var}  ·  n = {n}")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6 – Deployment
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🚀  Phase 6 · Deployment", expanded=True):
    st.markdown('<div class="phase-header">🔮 Predict & Export</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Single prediction")
        x_input = st.number_input("Enter x value", value=0.0, step=1.0,
                                  min_value=-100.0, max_value=100.0,
                                  key="pred_input")
        if st.button("▶  Predict"):
            x_arr = np.array([[x_input]])
            x_scaled = res["scaler"].transform(x_arr)
            y_hat = res["model"].predict(x_scaled)[0]
            y_true_approx = res["a_true"] * x_input + res["b_true"]
            st.success(f"**Predicted ŷ = {y_hat:.4f}**")
            st.caption(f"True line (no noise): {y_true_approx:.4f}  "
                       f"| Δ = {abs(y_hat - y_true_approx):.4f}")

    with c2:
        st.markdown("### Batch prediction")
        batch_text = st.text_area(
            "Enter x values (one per line)",
            value="10\n-20\n55\n-75\n0",
            height=140,
            key="batch_input",
        )
        if st.button("▶  Batch Predict"):
            try:
                xs = np.array([float(v) for v in batch_text.strip().split("\n")
                               if v.strip()]).reshape(-1, 1)
                xs_s = res["scaler"].transform(xs)
                preds = res["model"].predict(xs_s)
                batch_df = pd.DataFrame({"x": xs.ravel(), "ŷ": preds.round(4)})
                st.dataframe(batch_df, width="stretch", hide_index=True)
            except ValueError:
                st.error("Please enter valid numbers, one per line.")

    st.markdown("---")
    st.markdown("### Download model")
    c1, c2 = st.columns(2)
    with c1:
        # Save model + scaler to bytes buffer
        buf = io.BytesIO()
        joblib.dump({"model": res["model"], "scaler": res["scaler"]}, buf)
        buf.seek(0)
        st.download_button(
            label="⬇️  Download model (.joblib)",
            data=buf,
            file_name="linear_regression_model.joblib",
            mime="application/octet-stream",
        )
    with c2:
        csv_buf = res["df"].to_csv(index=False).encode()
        st.download_button(
            label="⬇️  Download dataset (.csv)",
            data=csv_buf,
            file_name="synthetic_data.csv",
            mime="text/csv",
        )

    st.markdown("""
---
**How to reload a saved model**
```python
import joblib, numpy as np
bundle = joblib.load("linear_regression_model.joblib")
model, scaler = bundle["model"], bundle["scaler"]
x_new = np.array([[42.0]])
y_pred = model.predict(scaler.transform(x_new))
print(y_pred)
```
""")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e3a5f;margin-top:30px;'>
<p style='text-align:center;color:#546e7a;font-size:0.78rem;'>
   CRISP-DM · Linear Regression Demo &nbsp;|&nbsp;
   Built with scikit-learn &amp; Streamlit &nbsp;|&nbsp;
   © 2025
</p>
""", unsafe_allow_html=True)
