"""
Hybrid Quantum-Classical ML Platform for Early Disease Detection
SIH26139 — Problem Statement 3

A Streamlit dashboard that benchmarks quantum-enhanced models against
classical baselines for binary disease classification.

Run: streamlit run app.py
"""

import os
import sys
import json
import numpy as np
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Project imports
from src.data.loader import load_default_dataset, load_csv_dataset
from src.data.preprocessing import preprocess, transform_new_data
from src.models.classical import get_classical_models
from src.models.quantum_vqc import HybridVQC
from src.models.quantum_kernel import QuantumKernelSVM
from src.evaluation.metrics import compute_metrics, compute_metrics_at_threshold, time_inference
from src.evaluation.explainability import (
    permutation_importance, draw_quantum_circuit, draw_kernel_circuit,
    compute_shap_explanations, compute_quantum_sensitivity,
    generate_plain_language_explanation
)

# ────────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Quantum ML Platform — Disease Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────
# Custom CSS for premium look
# ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #e0e0ff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8a8d0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(100,100,255,0.15);
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #7c83ff;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    .risk-low {
        background: linear-gradient(135deg, #0d7a3e, #15a050);
        color: #d0ffd0;
    }
    .risk-medium {
        background: linear-gradient(135deg, #b8860b, #d4a017);
        color: #fff8dc;
    }
    .risk-high {
        background: linear-gradient(135deg, #8b0000, #cc0000);
        color: #ffd0d0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-loaded {
        background: #0d3b1e;
        color: #4ade80;
        border: 1px solid #166534;
    }
    .status-training {
        background: #3b2d0d;
        color: #fbbf24;
        border: 1px solid #854d0e;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .clinician-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .admin-log-entry {
        padding: 0.4rem 0;
        border-bottom: 1px solid #2a2a4a;
        font-size: 0.85rem;
        color: #a8a8d0;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────
# Plotly theme
# ────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    'Logistic Regression': '#636EFA',
    'Random Forest': '#00CC96',
    'SVM (RBF)': '#AB63FA',
    'Quantum VQC': '#FFA15A',
    'Quantum SVM (QSVM)': '#FF6692',
}

# ────────────────────────────────────────────────────
# Session-based action log (for Admin view)
# ────────────────────────────────────────────────────
if 'action_log' not in st.session_state:
    st.session_state.action_log = []

def log_action(action: str):
    """Append a timestamped action to the in-session log."""
    st.session_state.action_log.append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'action': action,
    })

# ────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def results_exist():
    """Check if precomputed results are available."""
    return os.path.exists(os.path.join(RESULTS_DIR, 'metrics.json'))


@st.cache_resource
def load_precomputed_results():
    """Load precomputed results from the results/ directory."""
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'r') as f:
        data = json.load(f)
    
    preprocessing = joblib.load(os.path.join(RESULTS_DIR, 'preprocessing.pkl'))
    test_data = np.load(os.path.join(RESULTS_DIR, 'test_data.npz'))
    
    return {
        'results': data,
        'preprocessing': preprocessing,
        'X_train': test_data['X_train'],
        'X_test': test_data['X_test'],
        'y_train': test_data['y_train'],
        'y_test': test_data['y_test'],
    }


def get_risk_band(probability):
    """Map probability to risk band."""
    if probability < 0.3:
        return "Low", "risk-low"
    elif probability < 0.7:
        return "Medium", "risk-medium"
    else:
        return "High", "risk-high"


def render_metric_card(label, value, format_str=".3f"):
    """Render a styled metric card."""
    if isinstance(value, float):
        formatted = f"{value:{format_str}}"
    else:
        formatted = str(value)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{formatted}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def get_recommended_action(risk_label):
    """
    Return a placeholder recommended next step based on risk band.
    PLACEHOLDER RULE — not clinical guidance. Replace with clinician-validated
    rules before any real deployment.
    """
    actions = {
        "High": "⚠️ Recommend specialist referral for further diagnostic evaluation.",
        "Medium": "🔍 Recommend follow-up testing and monitoring.",
        "Low": "✅ Routine monitoring; no immediate action indicated.",
    }
    return actions.get(risk_label, "No recommendation available.")


# ────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # ── Role switcher ──
    st.divider()
    st.markdown("### 👤 Dashboard View")
    # DEMO-ONLY role switcher. This is NOT real access control (FR-9.x / production scope).
    # Do not mistake this for authentication or authorization gating.
    user_role = st.selectbox(
        "View as",
        ["Researcher", "Clinician", "Admin"],
        key="role_selector",
        help="Switch dashboard perspective for demo purposes. "
             "This is a demo-only switcher, not real access control."
    )
    
    st.divider()
    
    # Mode toggle
    demo_mode = st.toggle(
        "🚀 Quick Demo Mode",
        value=results_exist(),
        help="Load pre-computed results instantly. Disable for live training."
    )
    
    st.divider()
    
    # Dataset selection
    st.markdown("### 📁 Dataset")
    dataset_option = st.radio(
        "Choose dataset",
        ["🧬 Breast Cancer (built-in)", "📂 Upload CSV"],
        label_visibility="collapsed"
    )
    
    uploaded_file = None
    target_col = None
    if dataset_option == "📂 Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            target_col = st.selectbox(
                "Target column (binary label)",
                options=df_preview.columns.tolist(),
                index=len(df_preview.columns) - 1
            )
    
    st.divider()
    
    # Model hyperparameters (only show for Researcher role)
    if user_role == "Researcher":
        st.markdown("### 🔧 Hyperparameters")
        n_qubits = st.slider("Number of Qubits", 4, 8, 5, key="n_qubits_slider",
                             help="Also the number of PCA components")
        n_layers = st.slider("VQC Layers", 1, 4, 2, key="n_layers_slider")
        vqc_epochs = st.slider("VQC Epochs", 10, 100, 30, key="epochs_slider")
        qsvm_subsample = st.slider("QSVM Train Subsample", 50, 300, 100, key="subsample_slider",
                                   help="Reduce for faster kernel matrix computation")
    else:
        # Use defaults for non-Researcher roles
        n_qubits = 5
        n_layers = 2
        vqc_epochs = 30
        qsvm_subsample = 100
    
    st.divider()
    
    # Train button (only in live mode, Researcher role)
    train_clicked = False
    if not demo_mode and user_role == "Researcher":
        train_clicked = st.button("🚀 Train All Models", type="primary", use_container_width=True)
    
    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.75rem;'>"
        "SIH26139 / PS3<br>Hybrid Quantum ML Platform<br>for Early Disease Detection<br>"
        "<em>Team Peakso — Neutron</em></div>",
        unsafe_allow_html=True
    )


# ────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧬 Hybrid Quantum ML Platform</h1>
    <p>Early Disease Detection — Quantum vs. Classical Benchmarking</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────
# Data Loading
# ────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(dataset_key, _uploaded_file, target_col, n_qubits):
    """Load and preprocess the selected dataset."""
    if _uploaded_file is not None:
        X, y, feature_names, target_names = load_csv_dataset(_uploaded_file, target_col)
    else:
        X, y, feature_names, target_names = load_default_dataset()
    
    data = preprocess(X, y, n_qubits=n_qubits)
    return X, y, feature_names, target_names, data


# Create a stable key for caching
dataset_key = "breast_cancer" if uploaded_file is None else f"csv_{uploaded_file.name}"

try:
    X_raw, y_raw, feature_names, target_names, processed_data = load_and_preprocess(
        dataset_key, uploaded_file, target_col, n_qubits
    )
    log_action(f"Loaded dataset: {dataset_key}")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# ────────────────────────────────────────────────────
# Training / Loading Results
# ────────────────────────────────────────────────────
def train_all_models(processed_data, n_qubits, n_layers, vqc_epochs, qsvm_subsample):
    """Train all models and return results dict."""
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    all_results = {
        'metrics': {},
        'predictions': {},
        'probabilities': {},
        'train_times': {},
        'inference_times': {},
        'permutation_importance': {},
        'shap_importance': {},
        'quantum_sensitivity': {},
        'vqc_loss_history': [],
        'models': {},
    }
    
    progress_bar = st.progress(0, text="Training models...")
    status_text = st.empty()
    
    # Classical models
    classical_models = get_classical_models()
    total_models = len(classical_models) + 2  # +VQC +QSVM
    
    for i, (name, model) in enumerate(classical_models.items()):
        status_text.text(f"Training {name}...")
        t = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = compute_metrics(y_test, y_pred, y_proba)
        inf_time = time_inference(model.predict, X_test)
        
        all_results['metrics'][name] = metrics
        all_results['predictions'][name] = y_pred.tolist()
        all_results['probabilities'][name] = y_proba.tolist()
        all_results['train_times'][name] = t
        all_results['inference_times'][name] = inf_time
        all_results['models'][name] = model
        
        # Permutation importance
        pi = permutation_importance(
            model.predict, X_test, y_test,
            feature_names=processed_data['feature_names_pca'],
            n_repeats=10
        )
        all_results['permutation_importance'][name] = pi
        
        # SHAP importance
        try:
            shap_result = compute_shap_explanations(
                name, model, X_train, X_test,
                feature_names=processed_data['feature_names_pca']
            )
            all_results['shap_importance'][name] = shap_result['mean_abs_shap']
        except Exception:
            all_results['shap_importance'][name] = []
        
        progress_bar.progress((i + 1) / total_models, text=f"Trained {name}")
    
    # VQC
    status_text.text("Training Quantum VQC...")
    loss_placeholder = st.empty()
    
    vqc = HybridVQC(n_qubits=n_qubits, n_layers=n_layers, lr=0.01, epochs=vqc_epochs)
    
    loss_history_display = []
    def vqc_cb(epoch, loss):
        loss_history_display.append(loss)
        if (epoch + 1) % 2 == 0:
            progress_bar.progress(
                (len(classical_models) + (epoch + 1) / vqc_epochs) / total_models,
                text=f"VQC Epoch {epoch+1}/{vqc_epochs} — Loss: {loss:.4f}"
            )
    
    t = vqc.fit(X_train, y_train, progress_callback=vqc_cb)
    y_pred = vqc.predict(X_test)
    y_proba = vqc.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    inf_time = time_inference(vqc.predict, X_test)
    
    all_results['metrics']['Quantum VQC'] = metrics
    all_results['predictions']['Quantum VQC'] = y_pred.tolist()
    all_results['probabilities']['Quantum VQC'] = y_proba.tolist()
    all_results['train_times']['Quantum VQC'] = t
    all_results['inference_times']['Quantum VQC'] = inf_time
    all_results['vqc_loss_history'] = vqc.loss_history
    all_results['models']['Quantum VQC'] = vqc
    
    pi = permutation_importance(
        vqc.predict, X_test, y_test,
        feature_names=processed_data['feature_names_pca'],
        n_repeats=10
    )
    all_results['permutation_importance']['Quantum VQC'] = pi
    
    # Quantum sensitivity for VQC
    qs = compute_quantum_sensitivity(
        vqc.predict_proba, X_test,
        feature_names=processed_data['feature_names_pca'], delta=0.1
    )
    all_results['quantum_sensitivity']['Quantum VQC'] = qs
    
    progress_bar.progress((len(classical_models) + 1) / total_models, text="Trained VQC")
    
    # QSVM
    status_text.text(f"Training Quantum SVM (subsample={qsvm_subsample})...")
    qsvm = QuantumKernelSVM(n_qubits=n_qubits, subsample=qsvm_subsample)
    
    def qsvm_cb(step, total):
        pct = step / total
        progress_bar.progress(
            (len(classical_models) + 1 + pct) / total_models,
            text=f"QSVM Kernel: {pct:.0%}"
        )
    
    t = qsvm.fit(X_train, y_train, progress_callback=qsvm_cb)
    y_pred = qsvm.predict(X_test)
    y_proba = qsvm.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    inf_time = time_inference(qsvm.predict, X_test)
    
    all_results['metrics']['Quantum SVM (QSVM)'] = metrics
    all_results['predictions']['Quantum SVM (QSVM)'] = y_pred.tolist()
    all_results['probabilities']['Quantum SVM (QSVM)'] = y_proba.tolist()
    all_results['train_times']['Quantum SVM (QSVM)'] = t
    all_results['inference_times']['Quantum SVM (QSVM)'] = inf_time
    all_results['models']['Quantum SVM (QSVM)'] = qsvm
    
    pi = permutation_importance(
        qsvm.predict, X_test, y_test,
        feature_names=processed_data['feature_names_pca'],
        n_repeats=10
    )
    all_results['permutation_importance']['Quantum SVM (QSVM)'] = pi
    
    # Quantum sensitivity for QSVM
    qs = compute_quantum_sensitivity(
        qsvm.predict_proba, X_test,
        feature_names=processed_data['feature_names_pca'], delta=0.1
    )
    all_results['quantum_sensitivity']['Quantum SVM (QSVM)'] = qs
    
    progress_bar.progress(1.0, text="All models trained! ✅")
    status_text.empty()
    log_action("Trained all models (live training)")
    
    return all_results


# Determine what results to use
if demo_mode and results_exist():
    precomputed = load_precomputed_results()
    results = precomputed['results']
    X_test_eval = precomputed['X_test']
    y_test_eval = precomputed['y_test']
    X_train_eval = precomputed['X_train']
    y_train_eval = precomputed['y_train']
    st.markdown('<span class="status-badge status-loaded">✓ Pre-computed results loaded</span>',
                unsafe_allow_html=True)
    log_action("Loaded pre-computed results (Quick Demo Mode)")
elif train_clicked or ('trained_results' in st.session_state):
    if train_clicked:
        trained = train_all_models(processed_data, n_qubits, n_layers, vqc_epochs, qsvm_subsample)
        st.session_state['trained_results'] = trained
    
    trained = st.session_state['trained_results']
    results = trained
    X_test_eval = processed_data['X_test']
    y_test_eval = processed_data['y_test']
    X_train_eval = processed_data['X_train']
    y_train_eval = processed_data['y_train']
else:
    results = None
    X_test_eval = processed_data['X_test']
    y_test_eval = processed_data['y_test']
    X_train_eval = processed_data['X_train']
    y_train_eval = processed_data['y_train']


# ╔═══════════════════════════════════════════════════════╗
# ║                  RESEARCHER VIEW                       ║
# ╚═══════════════════════════════════════════════════════╝
if user_role == "Researcher":
    # ────────────────────────────────────────────────────
    # Tabs
    # ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Explorer",
        "🧠 Training",
        "📈 Benchmark",
        "🔮 Predict",
        "🔍 Explainability",
    ])
    log_action("Viewing Researcher dashboard")


    # ═══════════════════════════════════════════════════
    # TAB 1: Data Explorer
    # ═══════════════════════════════════════════════════
    with tab1:
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Samples", len(y_raw), "d")
        with col2:
            render_metric_card("Raw Features", X_raw.shape[1], "d")
        with col3:
            render_metric_card("PCA Components", n_qubits, "d")
        with col4:
            pca_total = sum(processed_data['pca_variance_ratio']) * 100
            render_metric_card("Variance Explained", f"{pca_total:.1f}%", "s")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Class Distribution")
            class_counts = pd.Series(y_raw).value_counts().sort_index()
            fig_class = go.Figure(data=[
                go.Bar(
                    x=[target_names[i] for i in class_counts.index],
                    y=class_counts.values,
                    marker_color=['#FF6692', '#636EFA'],
                    text=class_counts.values,
                    textposition='auto',
                )
            ])
            fig_class.update_layout(
                template=PLOTLY_TEMPLATE,
                height=350,
                margin=dict(t=30, b=30),
                xaxis_title="Class",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_class, use_container_width=True)
        
        with col_right:
            st.markdown("#### PCA Explained Variance")
            pca_var = processed_data['pca_variance_ratio']
            cumulative = np.cumsum(pca_var)
            
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(len(pca_var))],
                y=pca_var * 100,
                name='Individual',
                marker_color='#7c83ff',
                text=[f'{v*100:.1f}%' for v in pca_var],
                textposition='auto',
            ))
            fig_pca.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(len(pca_var))],
                y=cumulative * 100,
                name='Cumulative',
                line=dict(color='#FFA15A', width=3),
                mode='lines+markers',
            ))
            fig_pca.update_layout(
                template=PLOTLY_TEMPLATE,
                height=350,
                margin=dict(t=30, b=30),
                yaxis_title="Variance Explained (%)",
                showlegend=True,
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        
        # Feature stats
        st.markdown("#### Feature Statistics (Top 10)")
        df_stats = pd.DataFrame(X_raw[:, :10], columns=feature_names[:10])
        st.dataframe(df_stats.describe().round(3), use_container_width=True)
        
        # Correlation heatmap of PCA features
        st.markdown("#### PCA Feature Correlation")
        X_all_pca = np.vstack([processed_data['X_train'], processed_data['X_test']])
        corr = np.corrcoef(X_all_pca.T)
        labels = processed_data['feature_names_pca']
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr, x=labels, y=labels,
            colorscale='RdBu_r', zmid=0,
            text=np.round(corr, 2), texttemplate='%{text}',
        ))
        fig_corr.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_corr, use_container_width=True)


    # ═══════════════════════════════════════════════════
    # TAB 2: Training
    # ═══════════════════════════════════════════════════
    with tab2:
        st.markdown("### Model Training")
        
        # Cloud training warning
        if not demo_mode:
            st.warning(
                "⚠️ Live training may take longer on shared cloud hardware "
                "(HF Spaces CPU Basic). Quick Demo Mode is recommended for live demos."
            )
        
        if results is None:
            st.info("👈 Click **Train All Models** in the sidebar or enable **Quick Demo Mode** to see results.")
        else:
            # Mode indicator
            if demo_mode and results_exist():
                st.success("✅ Pre-computed results loaded instantly — no training required!")
            else:
                st.success("✅ Live training complete!")
            
            # Training times
            st.markdown("#### ⏱️ Training Times")
            train_times = results.get('train_times', {})
            
            cols = st.columns(len(train_times))
            for i, (name, t) in enumerate(train_times.items()):
                with cols[i]:
                    if t < 1:
                        render_metric_card(name, f"{t*1000:.0f}ms", "s")
                    else:
                        render_metric_card(name, f"{t:.1f}s", "s")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # VQC Loss curve
            loss_history = results.get('vqc_loss_history', [])
            if loss_history:
                st.markdown("#### 📉 VQC Training Loss")
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=list(range(1, len(loss_history) + 1)),
                    y=loss_history,
                    mode='lines',
                    line=dict(color='#FFA15A', width=2.5),
                    fill='tozeroy',
                    fillcolor='rgba(255,161,90,0.1)',
                ))
                fig_loss.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=350,
                    margin=dict(t=30, b=30),
                    xaxis_title="Epoch",
                    yaxis_title="BCE Loss",
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            # Model architecture info
            col_vqc, col_qsvm = st.columns(2)
            
            with col_vqc:
                st.markdown("#### 🔷 VQC Architecture")
                st.markdown(f"""
                ```
                Input ({n_qubits} features)
                → Linear({n_qubits}, {n_qubits}) + Tanh
                → AngleEmbedding(rotation='Y') on {n_qubits} qubits
                → StronglyEntanglingLayers × {n_layers}
                → [expval(PauliZ(i)) for i in {n_qubits}]
                → Linear({n_qubits}, 1) + Sigmoid
                ```
                """)
                st.caption(f"Optimizer: Adam (lr=0.01) | Loss: BCELoss | Epochs: {vqc_epochs}")
            
            with col_qsvm:
                st.markdown("#### 🔶 QSVM Architecture")
                st.markdown(f"""
                ```
                Fidelity Kernel k(x₁, x₂):
                → AngleEmbedding(x₁, rotation='Y')
                → adjoint(AngleEmbedding)(x₂, rotation='Y')
                → Measure P(|{'0'*n_qubits}⟩)
                
                Kernel Matrix → SVC(kernel='precomputed')
                ```
                """)
                st.caption(f"Qubits: {n_qubits} | Training subsample: {qsvm_subsample}")
            
            # Circuit diagrams
            st.markdown("#### 🔬 Quantum Circuit Diagrams")
            circ_col1, circ_col2 = st.columns(2)
            
            with circ_col1:
                try:
                    fig_vqc = draw_quantum_circuit(n_qubits=n_qubits, n_layers=n_layers)
                    st.pyplot(fig_vqc)
                    plt.close(fig_vqc)
                except Exception as e:
                    st.warning(f"Could not render VQC circuit: {e}")
            
            with circ_col2:
                try:
                    fig_kern = draw_kernel_circuit(n_qubits=n_qubits)
                    st.pyplot(fig_kern)
                    plt.close(fig_kern)
                except Exception as e:
                    st.warning(f"Could not render kernel circuit: {e}")


    # ═══════════════════════════════════════════════════
    # TAB 3: Benchmark
    # ═══════════════════════════════════════════════════
    with tab3:
        st.markdown("### Model Benchmarking — Quantum vs. Classical")
        
        if results is None:
            st.info("👈 Train models first to see benchmarks.")
        else:
            metrics_data = results.get('metrics', {})
            
            if not metrics_data:
                st.warning("No metrics available.")
            else:
                # Summary metrics table
                st.markdown("#### 📋 Performance Summary")
                
                metric_keys = ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'roc_auc']
                metric_labels = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'ROC-AUC']
                
                table_data = []
                for model_name in metrics_data:
                    m = metrics_data[model_name]
                    row = {'Model': model_name}
                    for key, label in zip(metric_keys, metric_labels):
                        row[label] = m.get(key, 0.0)
                    row['Train Time'] = results.get('train_times', {}).get(model_name, 0.0)
                    table_data.append(row)
                
                df_bench = pd.DataFrame(table_data)
                
                # Style the dataframe
                def highlight_best(s):
                    if s.name in ['Model', 'Train Time']:
                        return [''] * len(s)
                    is_best = s == s.max()
                    return ['background-color: #1a3a2a; font-weight: bold' if v else '' for v in is_best]
                
                styled = df_bench.style.apply(highlight_best).format({
                    'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Sensitivity': '{:.3f}',
                    'Specificity': '{:.3f}', 'F1': '{:.3f}', 'ROC-AUC': '{:.3f}',
                    'Train Time': '{:.3f}s'
                })
                st.dataframe(styled, use_container_width=True, hide_index=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Grouped bar chart
                col_bar, col_roc = st.columns(2)
                
                with col_bar:
                    st.markdown("#### 📊 Metrics Comparison")
                    fig_bar = go.Figure()
                    chart_metrics = ['accuracy', 'sensitivity', 'specificity', 'f1']
                    chart_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'F1']
                    
                    for model_name in metrics_data:
                        vals = [metrics_data[model_name].get(m, 0) for m in chart_metrics]
                        fig_bar.add_trace(go.Bar(
                            name=model_name,
                            x=chart_labels,
                            y=vals,
                            marker_color=COLORS.get(model_name, '#888'),
                            text=[f'{v:.3f}' for v in vals],
                            textposition='auto',
                        ))
                    
                    fig_bar.update_layout(
                        barmode='group',
                        template=PLOTLY_TEMPLATE,
                        height=450,
                        margin=dict(t=30, b=30),
                        legend=dict(orientation='h', y=-0.15),
                        yaxis_range=[0, 1.05],
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col_roc:
                    st.markdown("#### 📈 ROC Curves")
                    probabilities = results.get('probabilities', {})
                    
                    fig_roc = go.Figure()
                    
                    # Diagonal reference line
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='gray', dash='dash', width=1),
                        name='Random (AUC=0.5)',
                        showlegend=True,
                    ))
                    
                    from sklearn.metrics import roc_curve
                    for model_name in probabilities:
                        y_proba = np.array(probabilities[model_name])
                        fpr, tpr, _ = roc_curve(y_test_eval, y_proba)
                        auc_val = metrics_data[model_name].get('roc_auc', 0)
                        
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'{model_name} (AUC={auc_val:.3f})',
                            line=dict(color=COLORS.get(model_name, '#888'), width=2.5),
                        ))
                    
                    fig_roc.update_layout(
                        template=PLOTLY_TEMPLATE,
                        height=450,
                        margin=dict(t=30, b=30),
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        legend=dict(orientation='h', y=-0.2),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                # Training time comparison
                st.markdown("#### ⏱️ Training Time Comparison")
                train_times = results.get('train_times', {})
                
                fig_time = go.Figure(data=[
                    go.Bar(
                        x=list(train_times.keys()),
                        y=list(train_times.values()),
                        marker_color=[COLORS.get(n, '#888') for n in train_times.keys()],
                        text=[f'{t:.3f}s' for t in train_times.values()],
                        textposition='auto',
                    )
                ])
                fig_time.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=350,
                    margin=dict(t=30, b=30),
                    yaxis_title='Time (seconds)',
                    yaxis_type='log',
                )
                st.plotly_chart(fig_time, use_container_width=True)


    # ═══════════════════════════════════════════════════
    # TAB 4: Predict (Decision Support)
    # ═══════════════════════════════════════════════════
    with tab4:
        st.markdown("### 🔮 Disease Risk Prediction & Decision Support")
        
        if results is None:
            st.info("👈 Train models first to use prediction.")
        else:
            probabilities = results.get('probabilities', {})
            model_names = list(probabilities.keys())
            
            if not model_names:
                st.warning("No model probabilities available.")
            else:
                col_controls, col_viz = st.columns([1, 2])
                
                with col_controls:
                    st.markdown("#### Controls")
                    selected_model = st.selectbox(
                        "Select Model",
                        model_names,
                        key="predict_model"
                    )
                    
                    threshold = st.slider(
                        "Decision Threshold",
                        0.0, 1.0, 0.5, 0.01,
                        key="threshold_slider",
                        help="Adjust the classification threshold to balance sensitivity vs. specificity"
                    )
                    
                    st.markdown("---")
                    
                    # Single patient prediction
                    st.markdown("#### 🧑‍⚕️ Single Patient")
                    pred_mode = st.radio(
                        "Input mode",
                        ["Random test sample", "Manual input"],
                        key="pred_mode"
                    )
                    
                    if pred_mode == "Random test sample":
                        sample_idx = st.number_input(
                            "Sample index",
                            0, len(X_test_eval) - 1, 0,
                            key="sample_idx"
                        )
                        sample_features = X_test_eval[sample_idx]
                        sample_true_label = y_test_eval[sample_idx]
                    else:
                        st.caption(f"Enter {n_qubits} PCA components:")
                        sample_features = []
                        for i in range(n_qubits):
                            val = st.number_input(
                                f"PC{i+1}",
                                value=0.0, step=0.1,
                                key=f"manual_pc_{i}"
                            )
                            sample_features.append(val)
                        sample_features = np.array(sample_features)
                        sample_true_label = None
                
                with col_viz:
                    # Threshold analysis on full test set
                    y_proba = np.array(probabilities[selected_model])
                    threshold_metrics = compute_metrics_at_threshold(y_test_eval, y_proba, threshold)
                    
                    # Key metrics at threshold
                    st.markdown("#### Metrics at Current Threshold")
                    m_cols = st.columns(4)
                    with m_cols[0]:
                        render_metric_card("Sensitivity", threshold_metrics['sensitivity'])
                    with m_cols[1]:
                        render_metric_card("Specificity", threshold_metrics['specificity'])
                    with m_cols[2]:
                        render_metric_card("Precision", threshold_metrics['precision'])
                    with m_cols[3]:
                        render_metric_card("F1 Score", threshold_metrics['f1'])
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Confusion matrix + distribution
                    cm_col, dist_col = st.columns(2)
                    
                    with cm_col:
                        st.markdown("#### Confusion Matrix")
                        cm = np.array(threshold_metrics['confusion_matrix'])
                        cm_labels = target_names if target_names else ['Class 0', 'Class 1']
                        
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm, x=cm_labels, y=cm_labels,
                            colorscale='Blues',
                            text=cm, texttemplate='%{text}',
                            textfont=dict(size=20),
                            showscale=False,
                        ))
                        fig_cm.update_layout(
                            template=PLOTLY_TEMPLATE,
                            height=350,
                            margin=dict(t=30, b=30),
                            xaxis_title='Predicted',
                            yaxis_title='Actual',
                            yaxis=dict(autorange='reversed'),
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with dist_col:
                        st.markdown("#### Probability Distribution")
                        fig_dist = go.Figure()
                        
                        mask_0 = y_test_eval == 0
                        mask_1 = y_test_eval == 1
                        
                        fig_dist.add_trace(go.Histogram(
                            x=y_proba[mask_0], name=target_names[0] if target_names else 'Class 0',
                            marker_color='#FF6692', opacity=0.7, nbinsx=20,
                        ))
                        fig_dist.add_trace(go.Histogram(
                            x=y_proba[mask_1], name=target_names[1] if target_names else 'Class 1',
                            marker_color='#636EFA', opacity=0.7, nbinsx=20,
                        ))
                        fig_dist.add_vline(
                            x=threshold, line_dash="dash", line_color="yellow",
                            annotation_text=f"Threshold={threshold:.2f}",
                            annotation_position="top"
                        )
                        fig_dist.update_layout(
                            template=PLOTLY_TEMPLATE,
                            height=350,
                            margin=dict(t=30, b=30),
                            barmode='overlay',
                            xaxis_title='Predicted Probability',
                            yaxis_title='Count',
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Single patient result
                    st.markdown("#### 🧑‍⚕️ Patient Risk Assessment")
                    # Compute single-sample probability from stored proba
                    if pred_mode == "Random test sample":
                        patient_proba = y_proba[sample_idx]
                    else:
                        # For manual input, we need model prediction
                        patient_proba = 0.5  # Default
                        st.caption("⚠️ Manual input uses PCA-transformed features directly.")
                        # Try to get live prediction if we have trained models
                        if 'models' in results and selected_model in results['models']:
                            model_obj = results['models'][selected_model]
                            try:
                                p = model_obj.predict_proba(sample_features.reshape(1, -1))
                                patient_proba = p[0, 1]
                            except Exception:
                                pass
                    
                    risk_label, risk_class = get_risk_band(patient_proba)
                    
                    p_col1, p_col2, p_col3 = st.columns(3)
                    with p_col1:
                        render_metric_card("Probability", patient_proba)
                    with p_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">
                                <span class="risk-badge {risk_class}">{risk_label}</span>
                            </div>
                            <div class="metric-label">Risk Band</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with p_col3:
                        predicted_class = 1 if patient_proba >= threshold else 0
                        class_label = target_names[predicted_class] if target_names else f'Class {predicted_class}'
                        render_metric_card("Predicted Class", class_label, "s")
                    
                    if sample_true_label is not None:
                        true_label = target_names[sample_true_label] if target_names else f'Class {sample_true_label}'
                        match = "✅" if predicted_class == sample_true_label else "❌"
                        st.caption(f"True label: **{true_label}** {match}")
                    
                    log_action(f"Viewed prediction for sample (model={selected_model})")


    # ═══════════════════════════════════════════════════
    # TAB 5: Explainability
    # ═══════════════════════════════════════════════════
    with tab5:
        st.markdown("### 🔍 Model Explainability")
        
        if results is None:
            st.info("👈 Train models first to see explainability analysis.")
        else:
            perm_imp = results.get('permutation_importance', {})
            shap_imp = results.get('shap_importance', {})
            quant_sens = results.get('quantum_sensitivity', {})
            model_names_explain = list(perm_imp.keys())
            
            if not model_names_explain:
                st.warning("No explainability data available.")
            else:
                explain_model = st.selectbox(
                    "Select Model for Explanation",
                    model_names_explain,
                    key="explain_model"
                )
                
                is_quantum = ('Quantum' in explain_model or 'VQC' in explain_model or
                              'QSVM' in explain_model or 'Kernel' in explain_model)
                
                # ── Row 1: Permutation Importance + SHAP/Sensitivity ──
                col_imp, col_shap = st.columns(2)
                
                with col_imp:
                    st.markdown("#### 📊 Permutation Importance")
                    imp_data = perm_imp[explain_model]
                    
                    features = [d['feature'] for d in reversed(imp_data)]
                    means = [d['importance_mean'] for d in reversed(imp_data)]
                    stds = [d['importance_std'] for d in reversed(imp_data)]
                    
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Bar(
                        y=features,
                        x=means,
                        orientation='h',
                        marker_color='#7c83ff',
                        error_x=dict(type='data', array=list(reversed(stds)), visible=True),
                        text=[f'{m:.4f}' for m in means],
                        textposition='auto',
                    ))
                    fig_imp.update_layout(
                        template=PLOTLY_TEMPLATE,
                        height=max(300, len(features) * 50),
                        margin=dict(t=30, b=30, l=80),
                        xaxis_title='Mean Accuracy Decrease',
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    st.caption(
                        "**Permutation importance** measures how much accuracy drops when a feature "
                        "is randomly shuffled. Higher values = more important. This method is "
                        "model-agnostic and works identically for classical and quantum models."
                    )
                
                with col_shap:
                    if is_quantum:
                        # Quantum parameter sensitivity
                        st.markdown("#### ⚛️ Quantum Parameter Sensitivity")
                        sens_data = quant_sens.get(explain_model, [])
                        
                        if sens_data:
                            s_features = [d['feature'] for d in reversed(sens_data)]
                            s_means = [d['sensitivity_mean'] for d in reversed(sens_data)]
                            s_stds = [d['sensitivity_std'] for d in reversed(sens_data)]
                            
                            fig_sens = go.Figure()
                            fig_sens.add_trace(go.Bar(
                                y=s_features,
                                x=s_means,
                                orientation='h',
                                marker_color='#FFA15A',
                                error_x=dict(type='data', array=list(reversed(s_stds)), visible=True),
                                text=[f'{m:.4f}' for m in s_means],
                                textposition='auto',
                            ))
                            fig_sens.update_layout(
                                template=PLOTLY_TEMPLATE,
                                height=max(300, len(s_features) * 50),
                                margin=dict(t=30, b=30, l=80),
                                xaxis_title='Mean |ΔProbability|',
                            )
                            st.plotly_chart(fig_sens, use_container_width=True)
                            
                            st.caption(
                                "**Quantum parameter sensitivity** measures how much the output "
                                "probability changes when each input feature is perturbed by a small "
                                "delta (±0.1). This is NOT SHAP — it is a circuit-specific sensitivity "
                                "analysis for quantum models."
                            )
                        else:
                            st.info("Quantum sensitivity data not available. Run precompute or live training.")
                    else:
                        # SHAP for classical models
                        st.markdown("#### 🔬 SHAP Feature Importance")
                        shap_data = shap_imp.get(explain_model, [])
                        
                        if shap_data:
                            sh_features = [d['feature'] for d in reversed(shap_data)]
                            sh_means = [d['importance_mean'] for d in reversed(shap_data)]
                            
                            fig_shap = go.Figure()
                            fig_shap.add_trace(go.Bar(
                                y=sh_features,
                                x=sh_means,
                                orientation='h',
                                marker_color='#00CC96',
                                text=[f'{m:.4f}' for m in sh_means],
                                textposition='auto',
                            ))
                            fig_shap.update_layout(
                                template=PLOTLY_TEMPLATE,
                                height=max(300, len(sh_features) * 50),
                                margin=dict(t=30, b=30, l=80),
                                xaxis_title='Mean |SHAP Value|',
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                            
                            st.caption(
                                "**SHAP (SHapley Additive exPlanations)** assigns each feature a "
                                "contribution value based on cooperative game theory. Higher mean "
                                "|SHAP| = greater average impact on model output."
                            )
                        else:
                            st.info("SHAP data not available for this model. Run precompute or live training.")
                
                # ── Row 2: Circuit diagrams (quantum) or model descriptions (classical) ──
                st.markdown("---")
                st.markdown("#### 🔬 Model Details")
                
                if is_quantum and ('VQC' in explain_model or 'Quantum VQC' == explain_model):
                    try:
                        fig_circuit = draw_quantum_circuit(n_qubits=n_qubits, n_layers=n_layers)
                        st.pyplot(fig_circuit)
                        plt.close(fig_circuit)
                    except Exception as e:
                        st.warning(f"Could not render circuit: {e}")
                elif is_quantum and ('QSVM' in explain_model or 'Kernel' in explain_model):
                    try:
                        fig_circuit = draw_kernel_circuit(n_qubits=n_qubits)
                        st.pyplot(fig_circuit)
                        plt.close(fig_circuit)
                    except Exception as e:
                        st.warning(f"Could not render circuit: {e}")
                else:
                    if 'Logistic' in explain_model:
                        st.markdown("""
                        **Logistic Regression** uses a linear decision boundary with 
                        sigmoid activation. It's fast, interpretable, and serves as 
                        a strong baseline for binary classification.
                        """)
                    elif 'Random Forest' in explain_model:
                        st.markdown("""
                        **Random Forest** is an ensemble of decision trees that votes 
                        on the final prediction. It handles non-linear relationships 
                        and is robust to overfitting.
                        """)
                    elif 'SVM' in explain_model:
                        st.markdown("""
                        **SVM (RBF kernel)** maps data to a higher-dimensional space 
                        using the radial basis function kernel to find an optimal 
                        separating hyperplane.
                        """)
                
                # How it works section
                st.markdown("---")
                st.markdown("#### 💡 How Quantum Models Work")
                
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    st.markdown("""
                    **Variational Quantum Classifier (VQC)**
                    
                    1. Classical features are transformed via a linear layer and tanh activation
                    2. **Angle Embedding** encodes each feature as a Y-rotation on a qubit
                    3. **Strongly Entangling Layers** create quantum correlations between qubits
                       using parameterized rotation gates and CNOT entanglers
                    4. Measurements (expectation values of Pauli-Z) extract quantum information
                    5. A final classical layer maps to disease probability
                    
                    The quantum parameters are trained end-to-end via backpropagation,
                    leveraging quantum superposition and entanglement for feature extraction.
                    """)
                
                with exp_col2:
                    st.markdown("""
                    **Quantum Support Vector Machine (QSVM)**
                    
                    1. Each data point is encoded as a quantum state via Angle Embedding
                    2. The **fidelity kernel** measures similarity between two quantum states:
                       - Apply embedding of x₁, then the inverse embedding of x₂
                       - Probability of measuring |00...0⟩ = kernel value k(x₁, x₂)
                    3. This quantum kernel replaces the classical RBF kernel in an SVM
                    4. The quantum kernel can capture similarities that classical kernels miss
                    
                    The kernel leverages the exponentially large Hilbert space to compute 
                    inner products that would be intractable classically.
                    """)
                
                log_action(f"Viewed explainability for {explain_model}")


# ╔═══════════════════════════════════════════════════════╗
# ║                  CLINICIAN VIEW                        ║
# ╚═══════════════════════════════════════════════════════╝
elif user_role == "Clinician":
    st.markdown("### 🧑‍⚕️ Patient Risk Assessment")
    log_action("Viewing Clinician dashboard")
    
    if results is None:
        st.info("👈 Enable **Quick Demo Mode** in the sidebar to load results.")
    else:
        probabilities = results.get('probabilities', {})
        model_names = list(probabilities.keys())
        
        if not model_names:
            st.warning("No model results available.")
        else:
            # Use the best-performing model by default (highest accuracy)
            metrics_data = results.get('metrics', {})
            default_model = max(metrics_data, key=lambda m: metrics_data[m].get('accuracy', 0)) if metrics_data else model_names[0]
            
            # Simple sample selector
            st.markdown(
                '<div class="clinician-card">',
                unsafe_allow_html=True
            )
            
            col_select, col_result = st.columns([1, 2])
            
            with col_select:
                st.markdown("#### Select Patient Sample")
                sample_idx = st.number_input(
                    "Patient sample index",
                    0, len(X_test_eval) - 1, 0,
                    key="clinician_sample_idx",
                    help="Select a test patient to view risk assessment"
                )
                
                # Get probability
                y_proba = np.array(probabilities[default_model])
                patient_proba = y_proba[sample_idx]
                risk_label, risk_class = get_risk_band(patient_proba)
            
            with col_result:
                st.markdown("#### Risk Assessment")
                
                # Risk score display
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    render_metric_card("Risk Probability", patient_proba)
                with r_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">
                            <span class="risk-badge {risk_class}">{risk_label} Risk</span>
                        </div>
                        <div class="metric-label">Risk Band</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ── Plain-language explanation ──
            st.markdown(
                '<div class="clinician-card">',
                unsafe_allow_html=True
            )
            st.markdown("#### 📝 Explanation")
            
            # Get the appropriate importance data for the model
            shap_imp = results.get('shap_importance', {})
            quant_sens = results.get('quantum_sensitivity', {})
            perm_imp = results.get('permutation_importance', {})
            
            is_quantum = ('Quantum' in default_model or 'VQC' in default_model or
                          'QSVM' in default_model)
            
            if is_quantum and default_model in quant_sens:
                importances = quant_sens[default_model]
            elif not is_quantum and default_model in shap_imp and shap_imp[default_model]:
                importances = shap_imp[default_model]
            elif default_model in perm_imp:
                importances = perm_imp[default_model]
            else:
                importances = []
            
            explanation = generate_plain_language_explanation(
                importances, risk_label,
                feature_names_pca=processed_data.get('feature_names_pca', None)
            )
            
            st.markdown(f"> {explanation}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ── Recommended next step ──
            st.markdown(
                '<div class="clinician-card">',
                unsafe_allow_html=True
            )
            st.markdown("#### 🩺 Recommended Next Step")
            recommendation = get_recommended_action(risk_label)
            st.markdown(f"**{recommendation}**")
            st.caption(
                "⚠️ *Placeholder recommendation rule — not validated clinical guidance. "
                "In production, this should be replaced with clinician-approved decision protocols.*"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Model info footer
            st.caption(f"Assessment generated using: **{default_model}** "
                       f"(Accuracy: {metrics_data.get(default_model, {}).get('accuracy', 0):.1%})")
            
            log_action(f"Viewed clinician assessment for patient sample {sample_idx}")


# ╔═══════════════════════════════════════════════════════╗
# ║                    ADMIN VIEW                          ║
# ╚═══════════════════════════════════════════════════════╝
elif user_role == "Admin":
    st.markdown("### 🔧 Admin Dashboard")
    st.caption(
        "Demo-scope admin view — shows model status and session log. "
        "This is NOT a production audit trail (NFR-4 requires backend infrastructure "
        "this Streamlit prototype does not have)."
    )
    log_action("Viewing Admin dashboard")
    
    # ── Model Status ──
    st.markdown("#### 📦 Loaded Models & Status")
    
    if results is None:
        st.info("No models loaded. Enable Quick Demo Mode or train models.")
    else:
        model_status = []
        metrics_data = results.get('metrics', {})
        train_times = results.get('train_times', {})
        
        for model_name in metrics_data:
            acc = metrics_data[model_name].get('accuracy', 0.0)
            train_t = train_times.get(model_name, 0.0)
            
            # Try to get checkpoint timestamp
            timestamp = "Pre-computed"
            if demo_mode and results_exist():
                metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
                if os.path.exists(metrics_path):
                    mtime = os.path.getmtime(metrics_path)
                    timestamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            elif 'trained_results' in st.session_state:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " (this session)"
            
            model_status.append({
                'Model': model_name,
                'Status': '✅ Loaded',
                'Accuracy': f"{acc:.3f}",
                'Train Time': f"{train_t:.3f}s" if train_t < 60 else f"{train_t/60:.1f}min",
                'Last Trained': timestamp,
            })
        
        df_status = pd.DataFrame(model_status)
        st.dataframe(df_status, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ── Dataset Info ──
    st.markdown("#### 📊 Dataset Information")
    ds_col1, ds_col2, ds_col3 = st.columns(3)
    
    with ds_col1:
        render_metric_card("Dataset", dataset_key, "s")
    with ds_col2:
        render_metric_card("Samples", len(y_raw), "d")
    with ds_col3:
        render_metric_card("Features (raw)", X_raw.shape[1], "d")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    ds_col4, ds_col5, ds_col6 = st.columns(3)
    with ds_col4:
        render_metric_card("PCA Components", n_qubits, "d")
    with ds_col5:
        render_metric_card("Train Samples", len(processed_data['y_train']), "d")
    with ds_col6:
        render_metric_card("Test Samples", len(processed_data['y_test']), "d")
    
    st.markdown("---")
    
    # ── Session Action Log ──
    st.markdown("#### 📋 Session Action Log")
    st.caption("Demo-scope log — records actions taken in this session only. "
               "Not a production audit trail.")
    
    if st.session_state.action_log:
        log_entries = list(reversed(st.session_state.action_log))  # Most recent first
        for entry in log_entries[:50]:  # Cap at 50 entries
            st.markdown(
                f'<div class="admin-log-entry">'
                f'<code>{entry["time"]}</code> &nbsp; {entry["action"]}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No actions logged yet in this session.")
    
    st.markdown("---")
    
    # ── Configuration ──
    st.markdown("#### ⚙️ Current Configuration")
    config_data = {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'vqc_epochs': vqc_epochs,
        'qsvm_subsample': qsvm_subsample,
        'demo_mode': demo_mode,
        'results_available': results is not None,
    }
    st.json(config_data)
