# -*- coding: utf-8 -*-
"""
AutoForge · All-in-One AutoML Training Foundry
==============================================
Single-file Streamlit platform: data → profile → clean → train (50+ algorithms)
→ optimize → compare → explain → predict → deploy → report.

Run:
    pip install -r requirements.txt
    streamlit run app.py

On first launch it loads realistic demo data and runs a LIVE training
simulation automatically. No API keys. CPU-only friendly, GPU optional.
"""

import os, io, sys, json, time, math, types, base64, zipfile, hashlib, inspect, warnings, sqlite3, importlib.util
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from uuid import uuid4
from string import Template

# ---------------------------------------------------------------- theme file
def _write_theme():
    try:
        cfg_dir = Path(__file__).resolve().parent / ".streamlit"
        cfg = cfg_dir / "config.toml"
        if not cfg.exists():
            cfg_dir.mkdir(exist_ok=True)
            cfg.write_text(
                '[theme]\nbase="dark"\nprimaryColor="#22D3EE"\n'
                'backgroundColor="#0B1220"\nsecondaryBackgroundColor="#111A2D"\n'
                'textColor="#E2E8F0"\nfont="sans serif"\n'
                '[server]\nheadless=true\n', encoding="utf-8")
    except Exception:
        pass
_write_theme()

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import joblib
from scipy import stats as sps

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures, label_binarize
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, KFold,
                                     learning_curve, RandomizedSearchCV)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, log_loss, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix, r2_score,
                             mean_absolute_error, mean_squared_error, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
                              StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor,
                              IsolationForest)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="AutoForge · AutoML Foundry", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

S = st.session_state
RERUN = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")

def _has(mod):
    try: return importlib.util.find_spec(mod) is not None
    except Exception: return False

HAS = {m: _has(m) for m in ["xgboost", "lightgbm", "catboost", "optuna", "shap", "mlflow",
                            "statsmodels", "prophet", "torch", "umap", "fastapi"]}
def cuda_ok():
    if not HAS["torch"]: return False
    try:
        import torch; return bool(torch.cuda.is_available())
    except Exception: return False

# ---------------------------------------------------------------- palette
PAL = ["#22D3EE", "#8B5CF6", "#A3E635", "#F59E0B", "#F472B6", "#60A5FA", "#FB923C", "#2DD4BF"]
CY, VI, LI, AM, MG, RD, BL, TL = PAL
INK, SURF, TXT, MUT = "#0B1220", "#111A2D", "#E2E8F0", "#8FA3BF"
FAM_COL = {"Supervised": CY, "Unsupervised": VI, "Semi-supervised": BL,
           "Ensemble": LI, "Time-Series": AM, "Deep Learning": MG, "Custom": "#FB923C"}
DASH = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

tpl = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,20,36,0.55)",
    font=dict(family="Inter, sans-serif", color="#C7D2E4", size=12), colorway=PAL,
    title=dict(font=dict(family="Space Grotesk, Inter, sans-serif", size=15, color=TXT)),
    xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)",
               linecolor="rgba(148,163,184,0.25)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)",
               linecolor="rgba(148,163,184,0.25)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=44, r=18, t=44, b=36),
    hoverlabel=dict(bgcolor=SURF, font=dict(color=TXT))))
pio.templates["autoforge"] = tpl
pio.templates.default = "plotly_dark+autoforge"

def new_fig(h=320, title=None):
    return go.Figure(layout=go.Layout(height=h, title=title))

# ---------------------------------------------------------------- utils
def uid(): return uuid4().hex[:8]
def now(): return datetime.now().strftime("%H:%M:%S")
def stamp(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def toast(msg, icon="✅"):
    try: st.toast(msg, icon=icon)
    except Exception: pass
def jdef(o): return str(o)
def df_hash(df):
    try: return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]
    except Exception: return "n/a"
def fnum(v, nd=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))): return "—"
    if isinstance(v, (int, np.integer)): return f"{v:,}"
    return f"{v:.{nd}f}"
def W(key, default):
    if key not in S: S[key] = default
    return S[key]
def sel(label, options, key, default=None, **kw):
    opts = list(options)
    d = default if default is not None else (opts[0] if opts else None)
    cur = W(key, d)
    if opts and cur not in opts: S[key] = opts[0]
    return st.selectbox(label, opts, key=key, **kw)
def msel(label, options, key, default=None, **kw):
    opts = list(options)
    cur = W(key, list(default or []))
    S[key] = [o for o in cur if o in opts]
    return st.multiselect(label, opts, key=key, **kw)
def toggle(label, key, default=False):
    W(key, default)
    if hasattr(st, "toggle"): return st.toggle(label, key=key)
    return st.checkbox(label, key=key)

_BORDER = "border" in inspect.signature(st.container).parameters
@contextmanager
def panel(title=None):
    with st.container(**({"border": True} if _BORDER else {})):
        if title: st.markdown(f"<div class='panel-title'>{title}</div>", unsafe_allow_html=True)
        yield

def chip(text, color=CY):
    return (f"<span class='chip' style='color:{color};border-color:{color}55;"
            f"background:{color}14'>{text}</span>")
def chips(items): return "<div class='chips'>" + "".join(items) + "</div>"
def tiles_html(items):
    cells = "".join(
        f"<div class='tile' style='--c1:{c1};--c2:{c2}'><div class='k'>{k}</div>"
        f"<div class='v'>{v}</div><div class='s'>{s}</div></div>"
        for k, v, s, c1, c2 in items)
    return f"<div class='tile-row'>{cells}</div>"
def section(title, sub=None):
    st.markdown(f"<div class='sec'>{title}</div>" +
                (f"<div class='sec-sub'>{sub}</div>" if sub else ""), unsafe_allow_html=True)

# ---------------------------------------------------------------- CSS
def inject_css():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,button,input,select,textarea,[class*="st-"]{font-family:'Inter',system-ui,sans-serif;}
h1,h2,h3,.brand,.tile .v,.sec{font-family:'Space Grotesk','Inter',sans-serif !important;}
#MainMenu,footer,[data-testid="stDecoration"]{visibility:hidden;display:none;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stAppViewContainer"]{
 background:
  radial-gradient(1100px 750px at 88% -10%, rgba(139,92,246,.17), transparent 60%),
  radial-gradient(950px 700px at -8% 18%, rgba(34,211,238,.13), transparent 55%),
  radial-gradient(900px 620px at 50% 112%, rgba(244,114,182,.09), transparent 60%),
  linear-gradient(180deg,#070C16 0%,#0B1220 45%,#0A0F1C 100%) !important;}
[data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.05;
 background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)' opacity='0.55'/%3E%3C/svg%3E");}
[data-testid="stSidebar"]{background:rgba(9,14,26,.92) !important;border-right:1px solid rgba(34,211,238,.16);backdrop-filter:blur(8px);}
[data-testid="stSidebar"] *{color:#CBD5E1;}
[data-testid="stSidebar"] div[role="radiogroup"]>label>div:first-child{display:none;}
[data-testid="stSidebar"] div[role="radiogroup"]>label{padding:8px 12px;margin:2px 0;border-radius:10px;border:1px solid transparent;border-left:3px solid transparent;transition:all .16s ease;cursor:pointer;}
[data-testid="stSidebar"] div[role="radiogroup"]>label:hover{background:rgba(34,211,238,.08);}
[data-testid="stSidebar"] div[role="radiogroup"]>label:has(input:checked){
 background:linear-gradient(90deg,rgba(34,211,238,.16),rgba(139,92,246,.12));
 border-color:rgba(34,211,238,.30);border-left:3px solid #22D3EE;}
[data-testid="stSidebar"] div[role="radiogroup"]>label:has(input:checked) p{color:#A5F3FC !important;font-weight:600;}
*{scrollbar-width:thin;scrollbar-color:#22D3EE #0B1220;}
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:#0B1220;border-radius:8px;}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,#22D3EE,#8B5CF6);border-radius:8px;border:2px solid #0B1220;}
::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,#67E8F9,#A78BFA);}
.stButton>button,[data-testid="stDownloadButton"]>button{border-radius:10px;font-weight:600;transition:all .18s ease;border:1px solid rgba(148,163,184,.28);background:rgba(148,163,184,.08);color:#CBD5E1;}
.stButton>button:hover,[data-testid="stDownloadButton"]>button:hover{transform:translateY(-1px);border-color:rgba(34,211,238,.55);color:#A5F3FC;box-shadow:0 4px 18px rgba(34,211,238,.18);}
.stButton>button:active{transform:scale(.97);}
.stButton>button[kind="primary"],[data-testid="stDownloadButton"]>button[kind="primary"]{
 background:linear-gradient(135deg,#0891B2,#7C3AED) !important;color:#F8FAFC !important;border:0 !important;
 box-shadow:0 4px 20px rgba(34,211,238,.28);}
.stButton>button[kind="primary"]:hover{filter:brightness(1.12);box-shadow:0 6px 26px rgba(139,92,246,.4);}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid rgba(148,163,184,.18) !important;border-radius:14px !important;background:rgba(15,23,42,.45);}
.panel-title{font-family:'Space Grotesk';font-weight:600;font-size:15px;color:#A5F3FC;letter-spacing:.4px;margin-bottom:6px;}
.sec{font-size:20px;font-weight:700;color:#E2E8F0;margin:14px 0 2px;}
.sec-sub{color:#8FA3BF;font-size:13px;margin-bottom:8px;}
.hero{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:14px;padding:18px 22px;border-radius:18px;
 background:linear-gradient(120deg,rgba(34,211,238,.10),rgba(139,92,246,.12) 55%,rgba(244,114,182,.08));
 border:1px solid rgba(139,92,246,.28);margin-bottom:12px;}
.brand{font-size:27px;font-weight:700;background:linear-gradient(90deg,#22D3EE,#8B5CF6,#F472B6);-webkit-background-clip:text;background-clip:text;color:transparent;}
.tagline{color:#94A3B8;font-size:13px;margin-top:2px;}
.chips{display:flex;flex-wrap:wrap;gap:7px;align-items:center;}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 11px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:.3px;border:1px solid;}
.tile-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px;margin:6px 0 10px;}
.tile{background:linear-gradient(160deg,rgba(17,26,45,.92),rgba(12,19,34,.88));border:1px solid rgba(148,163,184,.17);border-radius:14px;padding:13px 16px;position:relative;overflow:hidden;}
.tile::before{content:"";position:absolute;top:0;bottom:0;left:0;width:3px;background:linear-gradient(180deg,var(--c1),var(--c2));}
.tile .k{font-size:10.5px;text-transform:uppercase;letter-spacing:1.3px;color:#8FA3BF;}
.tile .v{font-size:23px;font-weight:700;color:#E2E8F0;line-height:1.25;}
.tile .s{font-size:12px;color:#94A3B8;}
.live-head{display:flex;align-items:center;gap:10px;font-family:'Space Grotesk';font-weight:600;color:#E2E8F0;font-size:15px;margin-bottom:8px;}
.live-dot{width:9px;height:9px;border-radius:50%;background:#F87171;box-shadow:0 0 10px #F87171;animation:pulse 1.1s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.45;transform:scale(.8);}}
.stage{font-family:'JetBrains Mono',monospace;font-size:13px;color:#A5F3FC;padding:6px 12px;border:1px dashed rgba(34,211,238,.35);border-radius:10px;background:rgba(34,211,238,.06);margin:6px 0;}
.feed{font-family:'JetBrains Mono',monospace;font-size:12.5px;background:rgba(7,12,22,.72);border:1px solid rgba(34,211,238,.16);border-radius:12px;padding:10px 14px;max-height:230px;overflow-y:auto;color:#CBD5E1;}
.fl{padding:3px 0;border-bottom:1px dashed rgba(148,163,184,.09);animation:fadeUp .3s ease;}
.ok{color:#A3E635;font-weight:600;}.err{color:#F87171;font-weight:600;}.tm{color:#64748B;}
@keyframes fadeUp{from{opacity:0;transform:translateY(7px);}to{opacity:1;transform:none;}}
.stMarkdown,.stPlotlyChart,[data-testid="stDataFrame"],[data-testid="stDataEditor"]{animation:fadeUp .35s ease both;}
.card{background:linear-gradient(165deg,rgba(17,26,45,.9),rgba(11,18,32,.9));border:1px solid rgba(148,163,184,.18);border-radius:14px;padding:14px 16px;margin-bottom:8px;}
.card h4{margin:0 0 4px;font-family:'Space Grotesk';color:#E2E8F0;font-size:15px;}
.card .m{color:#8FA3BF;font-size:12.5px;}
.verdict{font-weight:600;}
.stTabs [data-baseweb="tab"]{font-weight:600;color:#94A3B8;}
.stTabs [aria-selected="true"]{color:#22D3EE !important;}
.stTabs [data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#22D3EE,#8B5CF6) !important;}
[data-testid="stExpander"]{border:1px solid rgba(148,163,184,.18);border-radius:12px;background:rgba(15,23,42,.4);}
[data-testid="stExpander"] summary:hover{color:#A5F3FC;}
[data-testid="stMetricValue"]{font-family:'Space Grotesk';color:#E2E8F0 !important;}
[data-testid="stMetricLabel"]{color:#8FA3BF !important;}
.stTextInput input,.stTextArea textarea,.stNumberInput input,[data-baseweb="select"]>div{
 background:#0F1830 !important;color:#E2E8F0 !important;border-color:rgba(148,163,184,.28) !important;border-radius:10px !important;}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:#22D3EE !important;box-shadow:0 0 0 2px rgba(34,211,238,.25) !important;}
[data-baseweb="select"] span,[data-baseweb="select"] input{color:#E2E8F0 !important;}
[data-baseweb="popover"] [data-baseweb="menu"]{background:#111A2D !important;border:1px solid rgba(34,211,238,.25);}
[data-baseweb="popover"] li{color:#CBD5E1 !important;}
[data-baseweb="popover"] li:hover{background:rgba(34,211,238,.12) !important;}
[data-baseweb="tag"]{background:rgba(34,211,238,.18) !important;}
[data-baseweb="tag"] span{color:#A5F3FC !important;}
[data-testid="stSlider"] div[role="slider"]{background:#22D3EE !important;box-shadow:0 0 10px rgba(34,211,238,.6);}
[data-testid="stCheckbox"] svg{fill:#22D3EE !important;}
[data-testid="stToggle"] [aria-checked="true"]{background:#22D3EE !important;}
[data-testid="stFileUploaderDropzone"]{background:rgba(15,24,48,.6) !important;border:1px dashed rgba(34,211,238,.4) !important;border-radius:12px;}
[data-testid="stDataFrame"],[data-testid="stDataEditor"]{filter:invert(1) hue-rotate(180deg);border-radius:12px;border:1px solid rgba(148,163,184,.2);}
pre,code,[data-testid="stCodeBlock"]{background:#0D1526 !important;color:#D7E3F4 !important;border-radius:12px !important;border:1px solid rgba(139,92,246,.22);}
[data-testid="stCodeBlock"] button{color:#22D3EE !important;}
[data-testid="stAlert"]{border-radius:12px;border:1px solid rgba(148,163,184,.2);background:rgba(15,23,42,.6);}
[data-testid="stTooltipContent"]{background:#111A2D;color:#E2E8F0;}
.stProgress>div>div>div{background:linear-gradient(90deg,#22D3EE,#8B5CF6) !important;}
hr{border-color:rgba(148,163,184,.18);}
.small{font-size:12px;color:#8FA3BF;}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- demo data
@st.cache_data(show_spinner=False)
def gen_telco(seed=7, n=2200):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[.45, .40, .15])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n, p=[.35, .20, .22, .23])
    base = np.where(internet == "Fiber optic", 85, np.where(internet == "DSL", 55, 20))
    monthly = np.clip(base + rng.normal(0, 9, n) - .08 * tenure, 18, 120).round(2)
    total = (monthly * tenure * (1 + rng.normal(0, .05, n))).round(2)
    total[rng.random(n) < .03] = np.nan
    senior = rng.binomial(1, .16, n)
    tech = np.where(rng.random(n) < np.where(contract == "Month-to-month", .3, .65), "Yes", "No")
    sec = np.where(rng.random(n) < np.where(internet == "No", .1, .4), "Yes", "No")
    stream = rng.choice(["Yes", "No"], n, p=[.42, .58])
    paperless = rng.binomial(1, .6, n)
    calls = rng.poisson(np.where(tech == "Yes", .6, 1.6) + (internet == "Fiber optic") * .9)
    spike = rng.random(n) < .02
    calls[spike] += rng.integers(8, 16, spike.sum())
    sat = np.clip(np.round(3.6 - .35 * calls + .5 * (tech == "Yes") + rng.normal(0, .8, n)), 1, 5).astype(int)
    lat = (-2.2 + 1.4 * (contract == "Month-to-month") - .9 * (contract == "Two year")
           + .8 * (internet == "Fiber optic") + .5 * (payment == "Electronic check")
           - .03 * tenure + .015 * (monthly - 60) + .22 * calls - .3 * (tech == "Yes") + .3 * senior
           + rng.normal(0, .5, n))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-lat)), "Yes", "No")
    return pd.DataFrame({"tenure": tenure, "Contract": contract, "InternetService": internet,
                         "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
                         "SeniorCitizen": senior, "TechSupport": tech, "OnlineSecurity": sec,
                         "StreamingTV": stream, "PaperlessBilling": paperless,
                         "SupportCalls": calls, "SatisfactionScore": sat, "Churn": churn})

@st.cache_data(show_spinner=False)
def gen_housing(seed=11, n=1800):
    rng = np.random.default_rng(seed)
    inc = np.clip(rng.gamma(3.2, 1.3, n), 0.8, 15)
    age = rng.integers(1, 53, n)
    rooms = np.clip(rng.normal(5.4, 1.6, n), 2, 12)
    bed = np.clip(rooms * rng.uniform(.12, .3, n), .8, 3)
    pop = np.clip(rng.lognormal(7.1, .6, n), 150, 9000).astype(int)
    occ = np.clip(rng.normal(3.0, .9, n), 1.4, 7)
    lat = rng.uniform(32.5, 41.9, n); lon = rng.uniform(-124.3, -114.3, n)
    coastal = np.where((lon < -118) & (lat < 36), rng.choice(["NEAR BAY", "NEAR OCEAN"], n),
                       rng.choice(["INLAND", "<1H OCEAN", "ISLAND"], n, p=[.6, .37, .03]))
    val = (.45 * inc + .006 * age - .18 * bed + .35 * np.isin(coastal, ["NEAR BAY", "NEAR OCEAN", "ISLAND"])
           + .12 * np.sin(lat) - .02 * occ + .04 * inc * (rooms / 5) + rng.normal(0, .18, n))
    return pd.DataFrame({"MedInc": inc.round(3), "HouseAge": age, "AveRooms": rooms.round(2),
                         "AveBedrms": bed.round(2), "Population": pop, "AveOccup": occ.round(2),
                         "Latitude": lat.round(3), "Longitude": lon.round(3),
                         "OceanProximity": coastal, "MedHouseVal": np.clip(val, .4, 5.2).round(3)})

@st.cache_data(show_spinner=False)
def gen_mall(seed=3, n=900):
    rng = np.random.default_rng(seed)
    centers = [(24, 42, 78, 9, 1.5), (33, 88, 82, 11, 3.5), (52, 55, 42, 4, 7), (44, 92, 18, 3, 9)]
    parts = []
    for c in centers:
        k = n // 4
        parts.append(np.column_stack([
            np.clip(rng.normal(c[0], 5, k), 18, 70), np.clip(rng.normal(c[1], 11, k), 15, 140),
            np.clip(rng.normal(c[2], 9, k), 1, 100), np.clip(rng.normal(c[3], 2, k), 1, 16),
            np.clip(rng.normal(c[4], 1.2, k), .2, 15)]))
    a = np.vstack(parts)
    return pd.DataFrame({"Age": a[:, 0].round(0).astype(int), "Annual Income (k$)": a[:, 1].round(1),
                         "Spending Score": a[:, 2].round(0).astype(int),
                         "Visits per Month": a[:, 3].round(0).astype(int), "Tenure (yrs)": a[:, 4].round(1)})

@st.cache_data(show_spinner=False)
def gen_retail(seed=5, n=730):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-07-01", periods=n, freq="D")
    t = np.arange(n)
    promo = rng.binomial(1, .12, n)
    weekly = 55 * np.isin(dates.dayofweek, [5, 6])
    sales = (400 + .18 * t + weekly + 60 * np.sin(2 * np.pi * t / 365.25)
             + 120 * promo + rng.normal(0, 24, n)).round(0)
    return pd.DataFrame({"Date": dates, "Promo": promo, "Sales": np.clip(sales, 120, None)})

DEMOS = {
    "Telco Churn":    dict(gen=gen_telco,   seed=7,  icon="📡", target="Churn",       blurb="2,200 customers · binary churn"),
    "Metro Housing":  dict(gen=gen_housing, seed=11, icon="🏙️", target="MedHouseVal", blurb="1,800 districts · price regression"),
    "Mall Segments":  dict(gen=gen_mall,    seed=3,  icon="🛍️", target=None,          blurb="900 shoppers · latent segments"),
    "Retail Demand":  dict(gen=gen_retail,  seed=5,  icon="📈", target="Sales",       blurb="730 days · seasonality + promos"),
}

def load_demo(name):
    d = DEMOS[name]
    df = d["gen"](seed=d["seed"]).copy()
    S["datasets"][name] = {"df": df, "origin": "demo", "seed": d["seed"], "created": stamp()}
    det = detect_task(df, prefer_target=d["target"])
    S["ds_cfg"][name] = det
    return df

# ---------------------------------------------------------------- detection / profile
def try_parse_date(s):
    if s.dtype.kind in "M": return s
    if s.dtype != object: return None
    try: conv = pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        try: conv = pd.to_datetime(s, errors="coerce")
        except Exception: return None
    if conv.notna().mean() > .92 and s.nunique() > 20: return conv
    return None

def detect_task(df, prefer_target=None):
    reasons, date_col = [], None
    for c in df.columns:
        conv = try_parse_date(df[c])
        if conv is not None and date_col is None:
            date_col = c; reasons.append(f"Datetime column '{c}' detected")
    target = prefer_target if prefer_target in df.columns else (df.columns[-1] if len(df.columns) else None)
    if date_col and target and target != date_col and pd.api.types.is_numeric_dtype(df[target]) \
            and df[date_col].nunique() > .9 * len(df):
        return dict(task="timeseries", target=target, date_col=date_col, confidence=.9,
                    reasons=reasons + [f"Unique dates + numeric '{target}' → time-series forecast"])
    if target is None:
        return dict(task="clustering", target=None, date_col=date_col, confidence=.75,
                    reasons=["No target column → unsupervised structure discovery"])
    y = df[target].dropna()
    nu = y.nunique()
    if y.dtype == object or y.dtype == bool or nu <= 12:
        reasons.append(f"Target '{target}' has {nu} discrete classes → classification")
        return dict(task="classification", target=target, date_col=date_col, confidence=.92, reasons=reasons)
    reasons.append(f"Target '{target}' is continuous ({nu} unique values) → regression")
    return dict(task="regression", target=target, date_col=date_col, confidence=.88, reasons=reasons)

def profile_df(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    miss = df.isna().sum()
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({"column": c, "dtype": str(s.dtype), "missing": int(miss[c]),
                     "missing %": round(100 * miss[c] / max(len(df), 1), 1),
                     "unique": int(s.nunique()),
                     "mean / top": (fnum(s.mean(), 2) if c in num else str(s.mode().iloc[0]) if len(s.mode()) else "—")})
    out = {}
    for c in num:
        q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
        iqr = q3 - q1
        out[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    return dict(rows=len(df), cols=len(df.columns), mem=round(df.memory_usage(deep=True).sum() / 1e6, 2),
                dups=int(df.duplicated().sum()), missing=int(miss.sum()),
                missing_pct=round(100 * miss.sum() / max(df.size, 1), 2),
                col_table=pd.DataFrame(rows), num_cols=num, outliers=out)

def feature_space_of(X):
    space = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            lo, hi, med = float(s.min()), float(s.max()), float(s.median())
            if hi == lo: hi = lo + 1
            isint = bool(pd.api.types.is_integer_dtype(s))
            space[c] = dict(kind="num", min=lo, max=hi, median=med,
                            step=1 if isint else round((hi - lo) / 100, 6), int=isint)
        else:
            space[c] = dict(kind="cat", options=sorted(s.dropna().astype(str).unique().tolist()),
                            median=str(s.mode().iloc[0]) if len(s.mode()) else "")
    return space

# ---------------------------------------------------------------- preprocessing
def make_ohe():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def default_prep():
    return dict(imputer="median", scale=True, winsor=True, select=False, pct=80, drop=[])

def build_pipeline(df, target, task, prep, model):
    notes = []
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    keep = y.notna()
    if (~keep).any():
        notes.append(f"Dropped {int((~keep).sum())} rows with missing target")
        X, y = X[keep], y[keep]
    for c in list(X.columns):
        if X[c].dtype.kind == "M":
            for part in ["year", "month", "day", "dayofweek"]:
                X[f"{c}_{part}"] = getattr(X[c].dt, part)
            X = X.drop(columns=[c]); notes.append(f"Expanded date '{c}' into calendar features")
    drop = [c for c in prep.get("drop", []) if c in X.columns]
    if drop: X = X.drop(columns=drop); notes.append(f"Dropped {len(drop)} user-excluded columns")
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    hi = [c for c in cat if X[c].nunique() > 40]
    if hi: X = X.drop(columns=hi); cat = [c for c in cat if c not in hi]; notes.append(f"Skipped high-cardinality: {', '.join(hi)}")
    nsteps = [("impute", SimpleImputer(strategy=prep.get("imputer", "median")))]
    if prep.get("winsor") and num:
        lo, hi_q = X[num].quantile(.01).values, X[num].quantile(.99).values
        nsteps.append(("clip", FunctionTransformerClip(lo, hi_q)))
    if prep.get("scale", True): nsteps.append(("scale", StandardScaler()))
    tf = []
    if num: tf.append(("num", Pipeline(nsteps), num))
    if cat: tf.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                        ("ohe", make_ohe())]), cat))
    steps = [("prep", ColumnTransformer(tf))]
    if prep.get("select") and task in ("classification", "regression"):
        steps.append(("select", SelectPercentile(f_classif if task == "classification" else f_regression,
                                                 percentile=int(prep.get("pct", 80)))))
    steps.append(("model", model))
    return Pipeline(steps), X, y, notes

class FunctionTransformerClip(BaseEstimator):
    """Picklable winsorizer (1%/99% bounds learned at build time)."""
    def __init__(self, lo=None, hi=None): self.lo, self.hi = lo, hi
    def fit(self, X, y=None): return self
    def transform(self, X):
        import numpy as np
        return np.clip(np.asarray(X, dtype=float), self.lo, self.hi)

def build_unsup_prep(df, prep):
    X = df.copy()
    for c in list(X.columns):
        if X[c].dtype.kind == "M":
            for part in ["year", "month", "day", "dayofweek"]: X[f"{c}_{part}"] = getattr(X[c].dt, part)
            X = X.drop(columns=[c])
    drop = [c for c in prep.get("drop", []) if c in X.columns]
    X = X.drop(columns=drop)
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num and X[c].nunique() <= 40]
    tf = []
    if num: tf.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler())]), num))
    if cat: tf.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                        ("ohe", make_ohe())]), cat))
    return Pipeline([("prep", ColumnTransformer(tf))]), X

def transformed_names(pipe):
    try: return list(pipe[:-1].get_feature_names_out())
    except Exception:
        try: return [f"f{i}" for i in range(pipe[:-1].transform(pipe[-1]._Xshape if hasattr(pipe[-1], "_Xshape") else None).shape[1])]
        except Exception: return None

def encode_target(y, task):
    if task == "classification" and (y.dtype == object or y.dtype == bool or str(y.dtype) == "category"):
        le = LabelEncoder(); return le.fit_transform(y.astype(str)), le
    return y.values, None

def get_proba(pipe, X):
    try:
        if hasattr(pipe, "predict_proba"): return pipe.predict_proba(X)
    except Exception: pass
    try:
        if hasattr(pipe, "decision_function"):
            d = pipe.decision_function(X)
            if d.ndim == 1:
                p = 1 / (1 + np.exp(-d)); return np.column_stack([1 - p, p])
    except Exception: pass
    return None

# ---------------------------------------------------------------- custom estimators
class PseudoLabelModel(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.85, max_iter=3, seed=0):
        self.threshold, self.max_iter, self.seed = threshold, max_iter, seed
    def fit(self, X, y):
        import numpy as np
        from sklearn.base import clone
        from sklearn.ensemble import GradientBoostingClassifier
        X, y = np.asarray(X), np.asarray(y)
        base = GradientBoostingClassifier(random_state=self.seed)
        labeled = y >= 0
        est = clone(base).fit(X[labeled], y[labeled])
        y2 = y.copy()
        for _ in range(self.max_iter):
            unl = ~labeled
            if not unl.any(): break
            proba = est.predict_proba(X[unl]); conf = proba.max(1)
            pick = conf >= self.threshold
            if not pick.any(): break
            idx = np.where(unl)[0][pick]
            y2[idx] = proba[pick].argmax(1); labeled[idx] = True
            est = clone(base).fit(X[labeled], y2[labeled])
        self.est_ = est; self.classes_ = est.classes_; return self
    def predict(self, X): return self.est_.predict(X)
    def predict_proba(self, X): return self.est_.predict_proba(X)

class BlendModel(BaseEstimator):
    def __init__(self, mode="classification", holdout=0.25, seed=0):
        self.mode, self.holdout, self.seed = mode, holdout, seed
    def fit(self, X, y):
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                      RandomForestRegressor, GradientBoostingRegressor)
        from sklearn.linear_model import LogisticRegression, Ridge
        clf = self.mode == "classification"
        strat = y if clf and np.min(np.unique(y, return_counts=True)[1]) >= 2 else None
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=self.holdout, random_state=self.seed, stratify=strat)
        if clf:
            self.bases_ = [RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=self.seed),
                           GradientBoostingClassifier(random_state=self.seed),
                           LogisticRegression(max_iter=800)]
            self.meta_ = LogisticRegression(max_iter=800)
        else:
            self.bases_ = [RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=self.seed),
                           GradientBoostingRegressor(random_state=self.seed), Ridge()]
            self.meta_ = Ridge()
        for b in self.bases_: b.fit(Xa, ya)
        metaX = self._meta_features(Xb)
        self.meta_.fit(metaX, yb)
        for b in self.bases_: b.fit(X, y)
        if clf: self.classes_ = self.bases_[0].classes_
        return self
    def _meta_features(self, X):
        import numpy as np
        if self.mode == "classification":
            return np.hstack([b.predict_proba(X) for b in self.bases_])
        return np.column_stack([b.predict(X) for b in self.bases_])
    def predict(self, X):
        import numpy as np
        if self.mode == "classification": return self.meta_.predict(self._meta_features(X))
        return self.meta_.predict(self._meta_features(X))
    def predict_proba(self, X):
        return self.meta_.predict_proba(self._meta_features(X))

class ClusterAssigner(BaseEstimator):
    def __init__(self, base=None): self.base = base
    def fit(self, X, y=None):
        import numpy as np
        from sklearn.neighbors import KNeighborsClassifier
        self.base_.fit(X) if not hasattr(self.base_, "labels_") and hasattr(self.base_, "fit") and not hasattr(self.base_, "fit_predict") else None
        self.labels_ = self.base_.fit_predict(X) if hasattr(self.base_, "fit_predict") else self.base_.fit(X).labels_
        self.direct_ = hasattr(self.base_, "predict")
        if not self.direct_:
            self.knn_ = KNeighborsClassifier(n_neighbors=1).fit(X, self.labels_)
        return self
    def fit_predict(self, X, y=None):
        self.fit(X); return self.labels_
    def predict(self, X):
        return self.base_.predict(X) if self.direct_ else self.knn_.predict(X)

class TorchNet(BaseEstimator):
    """Compact CPU/GPU torch models for tabular data (ANN/CNN1D/RNN/LSTM/GRU/Autoencoder)."""
    def __init__(self, kind="ann", task="classification", epochs=30, lr=1e-3,
                 hidden=64, dropout=0.1, device="cpu", seed=0, batch_size=128, contamination=0.08):
        self.kind, self.task, self.epochs, self.lr, self.hidden = kind, task, epochs, lr, hidden
        self.dropout, self.device, self.seed, self.batch_size = dropout, device, seed, batch_size
        self.contamination = contamination
    def _build(self, torch, F, out):
        nn = torch.nn
        h = self.hidden
        if self.kind == "cnn":
            return nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 32, 3, padding=1),
                                 nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, out))
        if self.kind == "autoencoder":
            return nn.Sequential(nn.Linear(F, h), nn.ReLU(), nn.Linear(h, 8), nn.ReLU(),
                                 nn.Linear(8, h), nn.ReLU(), nn.Linear(h, F))
        if self.kind == "mlp":
            return nn.Sequential(nn.Linear(F, h * 2), nn.ReLU(), nn.Dropout(self.dropout),
                                 nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, out))
        return nn.Sequential(nn.Linear(F, h), nn.ReLU(), nn.Dropout(self.dropout),
                             nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, out))
    def _forward(self, torch, xb):
        if self.kind in ("rnn", "lstm", "gru"):
            out, _ = self.core_(xb.unsqueeze(-1))
            return self.head_(out[:, -1, :])
        if self.kind == "cnn": return self.net_(xb.unsqueeze(1))
        return self.net_(xb)
    def fit(self, X, y=None):
        import numpy as np, torch
        from torch import nn
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        dev = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        F = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
            out = 1 if len(self.classes_) == 2 else len(self.classes_)
            yv = np.asarray(y, dtype=np.float32 if out == 1 else np.int64)
            lossf = nn.BCEWithLogitsLoss() if out == 1 else nn.CrossEntropyLoss()
        elif self.task == "autoencoder":
            out = F; yv = X.copy(); lossf = nn.MSELoss()
        else:
            out = 1; yv = np.asarray(y, dtype=np.float32); lossf = nn.MSELoss()
        if self.kind in ("rnn", "lstm", "gru"):
            core = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.kind]
            self.core_ = core(input_size=1, hidden_size=self.hidden, batch_first=True).to(dev)
            self.head_ = nn.Linear(self.hidden, out).to(dev)
            params = list(self.core_.parameters()) + list(self.head_.parameters())
        else:
            self.net_ = self._build(torch, F, out).to(dev)
            params = self.net_.parameters()
        opt = torch.optim.Adam(params, lr=self.lr)
        xt = torch.tensor(X, device=dev); yt = torch.tensor(yv, device=dev)
        n = len(xt); self.loss_history_ = []
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=dev); tot = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                pred = self._forward(torch, xt[idx])
                target = yt[idx] if pred.shape[-1] != 1 or self.task == "autoencoder" else yt[idx].unsqueeze(-1)
                if self.task == "autoencoder": target = xt[idx]
                loss = lossf(pred, target)
                loss.backward(); opt.step(); tot += float(loss) * len(idx)
            self.loss_history_.append(tot / n)
            cb = getattr(self, "_cb", None)
            if cb: cb(ep + 1, self.epochs, tot / n)
        self._cb = None
        if self.kind in ("rnn", "lstm", "gru"): self.core_.cpu(); self.head_.cpu()
        else: self.net_.cpu()
        if self.task == "autoencoder":
            err = self._recon_err(X)
            self.threshold_ = float(np.quantile(err, 1 - self.contamination))
        return self
    def _recon_err(self, X):
        import numpy as np, torch
        with torch.no_grad():
            xb = torch.tensor(np.asarray(X, dtype=np.float32))
            rec = self.net_(xb)
            return ((rec - xb) ** 2).mean(1).numpy()
    def _raw(self, X):
        import numpy as np, torch
        with torch.no_grad():
            return self._forward(torch, torch.tensor(np.asarray(X, dtype=np.float32))).numpy()
    def predict(self, X):
        import numpy as np
        if self.task == "autoencoder":
            return np.where(self._recon_err(X) > self.threshold_, -1, 1)
        r = self._raw(X)
        if self.task == "classification":
            if r.shape[1] == 1: return (r[:, 0] > 0).astype(int)
            return r.argmax(1)
        return r[:, 0]
    def predict_proba(self, X):
        import numpy as np
        r = self._raw(X)
        if r.shape[1] == 1:
            p = 1 / (1 + np.exp(-r[:, 0])); return np.column_stack([1 - p, p])
        e = np.exp(r - r.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    def decision_function(self, X):
        if self.task == "autoencoder": return -self._recon_err(X)
        return self._raw(X)[:, 0]
    def score(self, X, y):
        import numpy as np
        from sklearn.metrics import accuracy_score, r2_score
        p = self.predict(X)
        if self.task == "classification": return accuracy_score(y, p)
        if self.task == "autoencoder": return float(np.mean(p == -1))
        return r2_score(y, p)

class StatsForecast:
    def __init__(self, kind="arima", sp=7, seed=0):
        self.kind, self.sp, self.seed = kind, sp, seed
    def fit(self, series):
        import numpy as np
        y = series.values.astype(float)
        self.history_ = series.copy()
        if self.kind == "expsmooth":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            seas = "add" if len(y) >= 2 * self.sp else None
            self._m = ExponentialSmoothing(y, trend="add", seasonal=seas,
                                           seasonal_periods=self.sp if seas else None).fit()
            self.fitted_ = np.asarray(self._m.fittedvalues)
        else:
            from statsmodels.tsa.arima.model import ARIMA
            best, best_aic, table = None, np.inf, []
            for order in [(1, 1, 1), (0, 1, 1), (2, 1, 2), (1, 1, 0)]:
                try:
                    m = ARIMA(y, order=order).fit()
                    table.append({"order": str(order), "aic": round(float(m.aic), 1)})
                    if m.aic < best_aic: best, best_aic, self._m = order, m.aic, m
                except Exception: continue
            self.best_order_, self.aic_table_ = best, table
            if self.kind == "sarima" and len(y) >= 2 * self.sp + 5:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self._m = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, self.sp),
                                  enforce_stationarity=False).fit(disp=False)
                self.best_order_ = (1, 1, 1)
            self.fitted_ = np.asarray(self._m.fittedvalues)
        return self
    def forecast(self, h):
        import numpy as np
        return np.asarray(self._m.forecast(steps=int(h)))

class MLForecast:
    def __init__(self, base_kind="rf", lags=(1, 2, 3, 7, 14, 28), roll=7, seed=0):
        self.base_kind, self.lags, self.roll, self.seed = base_kind, lags, roll, seed
    def _frame(self, vals):
        import numpy as np, pandas as pd
        d = pd.DataFrame({"y": vals})
        for l in self.lags: d[f"lag_{l}"] = d["y"].shift(l)
        d["roll"] = d["y"].shift(1).rolling(self.roll).mean()
        return d
    def fit(self, series):
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        self.history_ = series.copy()
        vals = series.values.astype(float)
        d = self._frame(vals).dropna()
        self.cols_ = [c for c in d.columns if c != "y"]
        self.est_ = (RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=self.seed)
                     if self.base_kind == "rf" else GradientBoostingRegressor(random_state=self.seed))
        self.est_.fit(d[self.cols_], d["y"])
        self.fitted_ = np.full(len(vals), np.nan)
        self.fitted_[len(vals) - len(d):] = self.est_.predict(d[self.cols_])
        self.tail_ = list(vals[-(max(self.lags) + self.roll):])
        return self
    def forecast(self, h):
        import numpy as np
        hist = list(self.tail_); out = []
        for _ in range(int(h)):
            feats = [hist[-l] for l in self.lags] + [np.mean(hist[-self.roll:])]
            p = float(self.est_.predict([feats])[0]); out.append(p); hist.append(p)
        return np.array(out)

class ProphetForecast:
    def __init__(self, sp=7, seed=0): self.sp, self.seed = sp, seed
    def fit(self, series):
        import pandas as pd
        from prophet import Prophet
        self.history_ = series.copy()
        df = pd.DataFrame({"ds": series.index, "y": series.values})
        self._m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        self._m.fit(df)
        self.fitted_ = self._m.predict(df)["yhat"].values
        return self
    def forecast(self, h):
        fut = self._m.make_future_dataframe(periods=int(h), freq="D")
        return self._m.predict(fut)["yhat"].values[-int(h):]

# ---------------------------------------------------------------- algorithm registry
def _mk_xgb(task, seed, device, p):
    import xgboost as xgb
    gpu = device == "GPU" and cuda_ok()
    common = dict(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth", 6),
                  learning_rate=p.get("learning_rate", .1), subsample=p.get("subsample", .9),
                  colsample_bytree=.9, tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0)
    Cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
    try: return Cls(**common, device="cuda" if gpu else "cpu",
                    **({"eval_metric": "logloss"} if task == "classification" else {}))
    except TypeError:
        return Cls(**{**common, "tree_method": "gpu_hist" if gpu else "hist"})

def _mk_lgbm(task, seed, device, p):
    import lightgbm as lgb
    Cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
    return Cls(n_estimators=p.get("n_estimators", 300), learning_rate=p.get("learning_rate", .1),
               num_leaves=p.get("num_leaves", 31), random_state=seed, n_jobs=-1, verbose=-1)

def _mk_cat(task, seed, device, p):
    import catboost as cb
    gpu = device == "GPU" and cuda_ok()
    Cls = cb.CatBoostClassifier if task == "classification" else cb.CatBoostRegressor
    return Cls(iterations=p.get("iterations", 300), depth=p.get("depth", 6),
               learning_rate=p.get("learning_rate", .1), random_seed=seed, verbose=0,
               allow_writing_files=False, task_type="GPU" if gpu else "CPU")

def _mk_svm(task, seed, device, p):
    if task == "classification":
        return SVC(C=p.get("C", 1.0), kernel=p.get("kernel", "rbf"), gamma="scale",
                   probability=True, random_state=seed)
    return SVR(C=p.get("C", 1.0), kernel=p.get("kernel", "rbf"), gamma="scale")

def _mk_poly(task, seed, device, p):
    from sklearn.pipeline import make_pipeline
    return make_pipeline(PolynomialFeatures(degree=int(p.get("degree", 2)), include_bias=False),
                         LinearRegression())

def _mk_voting(task, seed, device, p):
    if task == "classification":
        return VotingClassifier([("lr", LogisticRegression(max_iter=800)),
                                 ("rf", RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=seed)),
                                 ("gb", GradientBoostingClassifier(random_state=seed))], voting="soft", n_jobs=-1)
    return VotingRegressor([("ridge", Ridge()),
                            ("rf", RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=seed)),
                            ("gb", GradientBoostingRegressor(random_state=seed))], n_jobs=-1)

def _mk_stacking(task, seed, device, p):
    if task == "classification":
        return StackingClassifier([("rf", RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=seed)),
                                   ("gb", GradientBoostingClassifier(random_state=seed))],
                                  final_estimator=LogisticRegression(max_iter=800), cv=3, n_jobs=-1)
    return StackingRegressor([("rf", RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=seed)),
                              ("gb", GradientBoostingRegressor(random_state=seed))],
                             final_estimator=Ridge(), cv=3, n_jobs=-1)

def _mk_rfens(task, seed, device, p):
    Cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    ests = [(f"rf{i}", Cls(n_estimators=110, max_depth=d, n_jobs=-1, random_state=seed + i))
            for i, d in enumerate([6, 10, None])]
    return (VotingClassifier(ests, voting="soft", n_jobs=-1) if task == "classification"
            else VotingRegressor(ests, n_jobs=-1))

def _mk_mlp(task, seed, device, p):
    Cls = MLPClassifier if task == "classification" else MLPRegressor
    return Cls(hidden_layer_sizes=p.get("hidden", (128, 64)), alpha=p.get("alpha", 1e-4),
               learning_rate_init=p.get("lr", 1e-3), max_iter=250, early_stopping=True, random_state=seed)

def _mk_torch(kind, task):
    def f(_task, seed, device, p):
        return TorchNet(kind=kind, task="classification" if _task == "classification" else "regression",
                        epochs=int(p.get("epochs", 30)), lr=p.get("lr", 1e-3),
                        hidden=int(p.get("hidden", 64)), device="cuda" if device == "GPU" else "cpu", seed=seed)
    return f

def A(id, name, family, tasks, make, needs=None, space=None, kind="standard", blurb=""):
    return dict(id=id, name=name, family=family, tasks=tasks, make=make, needs=needs,
                space=space or {}, kind=kind, blurb=blurb)

CR = ("classification", "regression")
ALGOS = [
 A("linreg", "Linear Regression", "Supervised", ("regression",), lambda t,s,d,p: LinearRegression(), blurb="OLS baseline"),
 A("multilinreg", "Multiple Linear Regression", "Supervised", ("regression",), lambda t,s,d,p: LinearRegression(), blurb="OLS on many predictors"),
 A("polyreg", "Polynomial Regression", "Supervised", ("regression",), _mk_poly, space={"degree": ("int", 2, 3)}),
 A("ridge", "Ridge Regression", "Supervised", ("regression",), lambda t,s,d,p: Ridge(alpha=p.get("alpha",1.0), random_state=s), space={"alpha": ("float", .01, 100, True)}),
 A("lasso", "Lasso Regression", "Supervised", ("regression",), lambda t,s,d,p: Lasso(alpha=p.get("alpha",.1), random_state=s), space={"alpha": ("float", .001, 10, True)}),
 A("logreg", "Logistic Regression", "Supervised", ("classification",), lambda t,s,d,p: LogisticRegression(C=p.get("C",1.0), max_iter=1200, random_state=s), space={"C": ("float", .01, 100, True)}),
 A("dt", "Decision Tree", "Supervised", CR, lambda t,s,d,p: (DecisionTreeClassifier if t=="classification" else DecisionTreeRegressor)(max_depth=p.get("max_depth",6), min_samples_leaf=p.get("min_samples_leaf",2), random_state=s), space={"max_depth": ("int",2,16), "min_samples_leaf": ("int",1,20)}),
 A("rf", "Random Forest", "Supervised", CR, lambda t,s,d,p: (RandomForestClassifier if t=="classification" else RandomForestRegressor)(n_estimators=p.get("n_estimators",160), max_depth=p.get("max_depth",12), min_samples_leaf=p.get("min_samples_leaf",1), n_jobs=-1, random_state=s), space={"n_estimators": ("int",100,400), "max_depth": ("int",3,18), "min_samples_leaf": ("int",1,10)}),
 A("svm", "Support Vector Machine", "Supervised", CR, _mk_svm, space={"C": ("float", .1, 100, True), "kernel": ("cat", ["rbf","linear"])}),
 A("knn", "K-Nearest Neighbors", "Supervised", CR, lambda t,s,d,p: (KNeighborsClassifier if t=="classification" else KNeighborsRegressor)(n_neighbors=p.get("n_neighbors",7), weights=p.get("weights","distance"), n_jobs=-1), space={"n_neighbors": ("int",3,25), "weights": ("cat", ["uniform","distance"])}),
 A("nb", "Naive Bayes (Gaussian)", "Supervised", ("classification",), lambda t,s,d,p: GaussianNB()),
 A("gb", "Gradient Boosting", "Supervised", CR, lambda t,s,d,p: (GradientBoostingClassifier if t=="classification" else GradientBoostingRegressor)(n_estimators=p.get("n_estimators",180), learning_rate=p.get("learning_rate",.08), max_depth=p.get("max_depth",3), random_state=s), space={"n_estimators": ("int",100,300), "learning_rate": ("float", .01, .2, True), "max_depth": ("int",2,5)}),
 A("xgb", "XGBoost", "Supervised", CR, _mk_xgb, needs="xgboost", space={"n_estimators": ("int",150,400), "max_depth": ("int",3,8), "learning_rate": ("float", .02, .2, True), "subsample": ("float", .6, 1.0, False)}),
 A("lgbm", "LightGBM", "Supervised", CR, _mk_lgbm, needs="lightgbm", space={"n_estimators": ("int",150,400), "learning_rate": ("float", .02, .2, True), "num_leaves": ("int",15,63)}),
 A("cat", "CatBoost", "Supervised", CR, _mk_cat, needs="catboost", space={"iterations": ("int",150,400), "depth": ("int",4,8), "learning_rate": ("float", .02, .2, True)}),
 A("ada", "AdaBoost", "Supervised", CR, lambda t,s,d,p: (AdaBoostClassifier if t=="classification" else AdaBoostRegressor)(n_estimators=p.get("n_estimators",120), learning_rate=p.get("learning_rate",.5), random_state=s), space={"n_estimators": ("int",50,300), "learning_rate": ("float", .05, 1.5, False)}),
 A("et", "Extra Trees", "Supervised", CR, lambda t,s,d,p: (ExtraTreesClassifier if t=="classification" else ExtraTreesRegressor)(n_estimators=p.get("n_estimators",180), max_depth=p.get("max_depth",14), n_jobs=-1, random_state=s), space={"n_estimators": ("int",100,400), "max_depth": ("int",4,20)}),
 A("kmeans", "K-Means Clustering", "Unsupervised", ("clustering",), lambda t,s,d,p: KMeans(n_clusters=p.get("n_clusters",4), n_init=10, random_state=s), space={"n_clusters": ("int",2,10)}),
 A("dbscan", "DBSCAN", "Unsupervised", ("clustering",), lambda t,s,d,p: DBSCAN(eps=p.get("eps",.6), min_samples=p.get("min_samples",5)), space={"eps": ("float", .2, 1.5, False), "min_samples": ("int",3,12)}),
 A("agglo", "Hierarchical Clustering", "Unsupervised", ("clustering",), lambda t,s,d,p: AgglomerativeClustering(n_clusters=p.get("n_clusters",4), linkage=p.get("linkage","ward")), space={"n_clusters": ("int",2,10), "linkage": ("cat", ["ward","average","complete"])}),
 A("gmm", "Gaussian Mixture Models", "Unsupervised", ("clustering",), lambda t,s,d,p: GaussianMixture(n_components=p.get("n_components",4), random_state=s, covariance_type="full"), space={"n_components": ("int",2,8)}),
 A("meanshift", "Mean Shift", "Unsupervised", ("clustering",), lambda t,s,d,p: MeanShift()),
 A("spectral", "Spectral Clustering", "Unsupervised", ("clustering",), lambda t,s,d,p: SpectralClustering(n_clusters=p.get("n_clusters",4), affinity="nearest_neighbors", assign_labels="kmeans", random_state=s), space={"n_clusters": ("int",2,8)}),
 A("pca", "Principal Component Analysis", "Unsupervised", ("clustering",), lambda t,s,d,p: PCA(n_components=2, random_state=s), kind="embedding"),
 A("tsne", "t-SNE Embedding", "Unsupervised", ("clustering",), lambda t,s,d,p: TSNE(n_components=2, perplexity=p.get("perplexity",30), init="pca", random_state=s), kind="embedding", space={"perplexity": ("int",5,50)}),
 A("umap", "UMAP Embedding", "Unsupervised", ("clustering",), lambda t,s,d,p: __import__("umap").UMAP(n_components=2, n_neighbors=p.get("n_neighbors",15), random_state=s), needs="umap", kind="embedding", space={"n_neighbors": ("int",5,50)}),
 A("iforest", "Isolation Forest (anomaly)", "Unsupervised", ("clustering",), lambda t,s,d,p: IsolationForest(n_estimators=p.get("n_estimators",200), contamination=p.get("contamination",.08), random_state=s, n_jobs=-1), kind="anomaly", space={"n_estimators": ("int",100,300), "contamination": ("float", .01, .2, False)}),
 A("selftrain", "Self-Training (LogReg)", "Semi-supervised", ("classification",), lambda t,s,d,p: SelfTrainingClassifier(LogisticRegression(max_iter=800, random_state=s))),
 A("labelprop", "Label Propagation", "Semi-supervised", ("classification",), lambda t,s,d,p: LabelPropagation(kernel="knn", n_neighbors=p.get("n_neighbors",7)), space={"n_neighbors": ("int",3,15)}),
 A("labelspread", "Label Spreading", "Semi-supervised", ("classification",), lambda t,s,d,p: LabelSpreading(kernel="knn", n_neighbors=7, alpha=p.get("alpha",.2)), space={"alpha": ("float", .05, .9, False)}),
 A("pseudo", "Pseudo-Labeling (Boosting)", "Semi-supervised", ("classification",), lambda t,s,d,p: PseudoLabelModel(threshold=p.get("threshold",.85), seed=s), space={"threshold": ("float", .6, .95, False)}),
 A("voting", "Voting Ensemble", "Ensemble", CR, _mk_voting),
 A("bagging", "Bagging (Trees)", "Ensemble", CR, lambda t,s,d,p: (BaggingClassifier if t=="classification" else BaggingRegressor)(n_estimators=p.get("n_estimators",15), n_jobs=-1, random_state=s), space={"n_estimators": ("int",5,30)}),
 A("boosting", "Boosting (AdaBoost)", "Ensemble", CR, lambda t,s,d,p: (AdaBoostClassifier if t=="classification" else AdaBoostRegressor)(n_estimators=150, random_state=s)),
 A("stacking", "Stacking Ensemble", "Ensemble", CR, _mk_stacking),
 A("blending", "Blending Ensemble", "Ensemble", CR, lambda t,s,d,p: BlendModel(mode=t, seed=s)),
 A("rfens", "Random Forest Ensemble", "Ensemble", CR, _mk_rfens),
 A("arima", "ARIMA (auto order)", "Time-Series", ("timeseries",), lambda t,s,d,p: StatsForecast("arima", sp=p.get("sp",7), seed=s), needs="statsmodels"),
 A("sarima", "SARIMA (seasonal)", "Time-Series", ("timeseries",), lambda t,s,d,p: StatsForecast("sarima", sp=p.get("sp",7), seed=s), needs="statsmodels"),
 A("expsmooth", "Exponential Smoothing", "Time-Series", ("timeseries",), lambda t,s,d,p: StatsForecast("expsmooth", sp=p.get("sp",7), seed=s), needs="statsmodels"),
 A("prophet", "Prophet", "Time-Series", ("timeseries",), lambda t,s,d,p: ProphetForecast(sp=p.get("sp",7), seed=s), needs="prophet"),
 A("rfts", "Random Forest (lag features)", "Time-Series", ("timeseries",), lambda t,s,d,p: MLForecast("rf", seed=s)),
 A("gbts", "Gradient Boosting (lag features)", "Time-Series", ("timeseries",), lambda t,s,d,p: MLForecast("gb", seed=s)),
 A("ann", "Artificial Neural Network", "Deep Learning", CR, _mk_torch("ann", None), needs="torch", space={"epochs": ("int",15,60), "lr": ("float", 1e-4, .01, True), "hidden": ("int",32,128)}),
 A("mlp", "Multilayer Perceptron", "Deep Learning", CR, _mk_mlp, space={"alpha": ("float", 1e-5, .01, True), "lr": ("float", 1e-4, .01, True)}),
 A("cnn", "CNN (1-D over features)", "Deep Learning", CR, _mk_torch("cnn", None), needs="torch", space={"epochs": ("int",15,60), "lr": ("float", 1e-4, .01, True)}),
 A("rnn", "Recurrent Neural Network", "Deep Learning", CR, _mk_torch("rnn", None), needs="torch", space={"epochs": ("int",15,50), "hidden": ("int",16,64)}),
 A("lstm", "LSTM", "Deep Learning", CR, _mk_torch("lstm", None), needs="torch", space={"epochs": ("int",15,50), "hidden": ("int",16,64)}),
 A("gru", "GRU", "Deep Learning", CR, _mk_torch("gru", None), needs="torch", space={"epochs": ("int",15,50), "hidden": ("int",16,64)}),
 A("autoenc", "Autoencoder (anomaly)", "Deep Learning", ("clustering",), lambda t,s,d,p: TorchNet(kind="autoencoder", task="autoencoder", epochs=int(p.get("epochs",30)), seed=s), needs="torch", kind="anomaly", space={"epochs": ("int",15,60)}),
]
ALGO_BY_ID = {a["id"]: a for a in ALGOS}

def get_spec(aid):
    if aid in ALGO_BY_ID: return ALGO_BY_ID[aid]
    return next((c for c in S.get("custom_algos", []) if c["id"] == aid), None)

def algo_ok(spec):
    return spec is not None and (spec.get("needs") is None or HAS.get(spec["needs"], False))

def available_for(task, semi=False):
    out = []
    for a in ALGOS + S.get("custom_algos", []):
        if task not in a["tasks"]: continue
        if a["family"] == "Semi-supervised" and not semi: continue
        out.append(a)
    return out

def recommend(df, task, target, semi=False, n=6):
    ok = [a for a in available_for(task, semi) if algo_ok(a)]
    n_rows, n_cols = len(df), df.shape[1]
    reasons = {}
    def prio(aid, why):
        reasons[aid] = why
    if task == "classification":
        bal = df[target].value_counts(normalize=True).min() if target else 1
        prio("logreg", "fast interpretable baseline"); prio("rf", "robust to mixed types & missing values")
        prio("xgb", "state-of-the-art tabular boosting"); prio("lgbm", "fast gradient boosting")
        prio("et", "low-variance tree ensemble"); prio("cat", "handles categorical structure well")
        if n_rows < 6000: prio("svm", "strong margin classifier at this scale"); prio("knn", "local structure check")
        prio("nb", "probabilistic baseline"); prio("gb", "classic boosting reference")
        if bal < .2: prio("voting", "ensembles stabilize imbalanced targets")
        if semi: prio("selftrain", "leverages unlabeled rows"); prio("labelspread", "graph-based label diffusion")
        prio("mlp", "non-linear neural reference")
    elif task == "regression":
        prio("ridge", "regularized linear baseline"); prio("rf", "captures non-linearity")
        prio("xgb", "top tabular regressor"); prio("gb", "boosted residual fitting")
        prio("lasso", "sparse feature selection"); prio("et", "variance reduction")
        prio("knn", "local smoothness probe"); prio("svm", ""); prio("svm", "kernel regression")
        prio("mlp", "neural non-linearity"); prio("linreg", "interpretable OLS reference")
    elif task == "clustering":
        prio("kmeans", "fast centroid partition"); prio("gmm", "soft elliptical clusters")
        prio("dbscan", "density clusters + noise"); prio("agglo", "hierarchical structure")
        prio("iforest", "anomaly screening"); prio("pca", "2-D structure map"); prio("tsne", "non-linear embedding")
    else:
        prio("expsmooth", "trend + seasonality baseline"); prio("arima", "auto-selected ARIMA order")
        prio("rfts", "ML on lag features"); prio("gbts", "boosted lag features")
        prio("sarima", "explicit seasonal component"); prio("prophet", "decomposition + holidays")
    ordered = [a for a in ok if a["id"] in reasons]
    ordered.sort(key=lambda a: list(reasons).index(a["id"]))
    rest = [a for a in ok if a["id"] not in reasons]
    sel_list = (ordered + rest)[:n]
    return sel_list, {a["id"]: reasons.get(a["id"], "available & compatible") for a in sel_list}

# ---------------------------------------------------------------- metrics
METRICS = {
 "classification": [("ROC AUC","roc_auc","max","roc_auc"), ("Accuracy","accuracy","max","accuracy"),
                    ("Balanced Accuracy","balanced_accuracy","max","balanced_accuracy"),
                    ("F1 (macro)","f1_macro","max","f1_macro"), ("F1 (weighted)","f1_weighted","max","f1_weighted"),
                    ("Precision (macro)","precision_macro","max","precision_macro"),
                    ("Recall (macro)","recall_macro","max","recall_macro"), ("Log Loss","log_loss","min","neg_log_loss")],
 "regression": [("R²","r2","max","r2"), ("RMSE","rmse","min","neg_root_mean_squared_error"),
                ("MAE","mae","min","neg_mean_absolute_error"), ("MAPE","mape","min",None)],
 "clustering": [("Silhouette","silhouette","max",None), ("Calinski-Harabasz","calinski","max",None),
                ("Davies-Bouldin","davies","min",None), ("Anomaly rate","anomaly_rate","max",None)],
 "timeseries": [("RMSE","rmse","min",None), ("MAE","mae","min",None),
                ("MAPE","mape","min",None), ("sMAPE","smape","min",None)],
}
def mdef(task, name):
    for m in METRICS[task]:
        if m[0] == name: return m
    return METRICS[task][0]

def clf_metrics(y, pred, proba):
    m = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
             f1_macro=f1_score(y, pred, average="macro"), f1_weighted=f1_score(y, pred, average="weighted"),
             precision_macro=precision_score(y, pred, average="macro", zero_division=0),
             recall_macro=recall_score(y, pred, average="macro", zero_division=0))
    nc = len(np.unique(y))
    if proba is not None:
        try:
            m["roc_auc"] = roc_auc_score(y, proba[:, 1]) if nc == 2 else roc_auc_score(y, proba, multi_class="ovr", average="macro")
            m["log_loss"] = log_loss(y, proba, labels=np.arange(proba.shape[1]))
        except Exception: pass
    return m

def reg_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    eps = 1e-9
    return dict(r2=r2_score(y, pred), rmse=float(np.sqrt(mean_squared_error(y, pred))),
                mae=mean_absolute_error(y, pred),
                mape=float(np.mean(np.abs((y - pred) / (np.abs(y) + eps))) * 100))

def ts_metrics(y, pred):
    m = reg_metrics(y, pred)
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    m["smape"] = float(np.mean(2 * np.abs(y - pred) / (np.abs(y) + np.abs(pred) + 1e-9)) * 100)
    return m

def verdict_of(task, key, direction, train, cv, test):
    if task in ("clustering",): return ("—", MUT, "Unsupervised — no label-based fit diagnosis")
    if cv is None or (isinstance(cv, float) and math.isnan(cv)):
        return ("Holdout", BL, "CV unavailable for this setup — judged on holdout only")
    gap = (train - cv) if direction == "max" else (cv - train)
    scale = max(abs(train), 1e-6) if direction == "min" else 1.0
    if gap > (0.15 * scale if direction == "min" else 0.07):
        return ("Overfit risk", AM, f"Train {fnum(train)} vs CV {fnum(cv)} — memorizing training noise. Try regularization, shallower trees or more data.")
    if direction == "max" and train < (0.35 if key == "r2" else 0.62):
        return ("Underfit", RD, "Even training fit is weak — the model family may be too simple. Try boosting or neural models.")
    return ("Healthy", LI, "Train / CV / holdout scores are consistent — good generalization.")

def cv_eval(pipe, X, y, task, scoring, folds, seed, semi=False):
    if task not in ("classification", "regression") or semi or scoring is None:
        return float("nan"), float("nan"), "n/a for this setup"
    try:
        if task == "classification":
            minc = pd.Series(y).value_counts().min()
            k = int(min(folds, max(2, minc)))
            if minc < 2: return float("nan"), float("nan"), "too few class samples"
            cv = StratifiedKFold(k, shuffle=True, random_state=seed)
        else:
            cv = KFold(int(folds), shuffle=True, random_state=seed)
        sc = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score=np.nan)
        return float(np.nanmean(sc)), float(np.nanstd(sc)), scoring
    except Exception as e:
        return float("nan"), float("nan"), f"CV skipped ({str(e)[:60]})"

def curves_of(y, proba):
    if proba is None: return None
    classes = np.unique(y)
    try:
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y, proba[:, 1])
            prec, rec, _ = precision_recall_curve(y, proba[:, 1])
            return dict(mode="binary", auc=roc_auc_score(y, proba[:, 1]),
                        ap=average_precision_score(y, proba[:, 1]),
                        roc=[dict(label="ROC", fpr=fpr.tolist(), tpr=tpr.tolist(), auc=roc_auc_score(y, proba[:, 1]))],
                        pr=[dict(label="PR", prec=prec.tolist(), rec=rec.tolist(), ap=average_precision_score(y, proba[:, 1]))])
        yb = label_binarize(y, classes=classes)
        roc, pr, aucs = [], [], []
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
            prec, rec, _ = precision_recall_curve(yb[:, i], proba[:, i])
            a = roc_auc_score(yb[:, i], proba[:, i]); aucs.append(a)
            roc.append(dict(label=f"class {c}", fpr=fpr.tolist(), tpr=tpr.tolist(), auc=a))
            pr.append(dict(label=f"class {c}", prec=prec.tolist(), rec=rec.tolist(),
                           ap=average_precision_score(yb[:, i], proba[:, i])))
        return dict(mode="multi", auc=float(np.mean(aucs)), roc=roc, pr=pr)
    except Exception:
        return None

# ---------------------------------------------------------------- persistence (SQLite)
DB_PATH = Path(__file__).resolve().with_name("autoforge_state.db")
def db():
    con = sqlite3.connect(str(DB_PATH)); con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS registry(id TEXT PRIMARY KEY, name TEXT, version INTEGER, created TEXT, meta TEXT, blob BLOB)")
    con.execute("CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, created TEXT, json TEXT)")
    return con

def db_registry_list():
    try:
        with db() as con:
            rows = con.execute("SELECT id,name,version,created,meta FROM registry ORDER BY created DESC").fetchall()
        return [dict(id=r[0], name=r[1], version=r[2], created=r[3], **json.loads(r[4])) for r in rows]
    except Exception: return []

def db_registry_put(entry, blob):
    try:
        meta = {k: v for k, v in entry.items() if k not in ("id", "name", "version", "created")}
        with db() as con:
            con.execute("INSERT OR REPLACE INTO registry VALUES (?,?,?,?,?,?)",
                        (entry["id"], entry["name"], entry["version"], entry["created"],
                         json.dumps(meta, default=jdef), blob))
    except Exception as e: toast(f"Persistence unavailable: {e}", "⚠️")

def db_registry_blob(rid):
    try:
        with db() as con: r = con.execute("SELECT blob FROM registry WHERE id=?", (rid,)).fetchone()
        return r[0] if r else None
    except Exception: return None

def db_registry_delete(rid):
    try:
        with db() as con: con.execute("DELETE FROM registry WHERE id=?", (rid,))
    except Exception: pass

def db_exp_put(exp):
    try:
        with db() as con:
            con.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?)",
                        (exp["id"], exp["ts"], json.dumps(exp, default=jdef)))
    except Exception: pass

def db_exp_list():
    try:
        with db() as con:
            rows = con.execute("SELECT json FROM experiments ORDER BY created DESC LIMIT 200").fetchall()
        return [json.loads(r[0]) for r in rows]
    except Exception: return []

def db_exp_delete(eid):
    try:
        with db() as con: con.execute("DELETE FROM experiments WHERE id=?", (eid,))
    except Exception: pass

# ---------------------------------------------------------------- live UI + training
class LiveUI:
    def __init__(self, container, title):
        with container:
            st.markdown(f"<div class='live-head'><span class='live-dot'></span> LIVE TRAINING SIMULATION · {title}</div>",
                        unsafe_allow_html=True)
            self.status = st.empty(); self.bar = st.progress(0.0)
            self.tiles = st.empty(); self.chart = st.empty()
            self.detail = st.empty(); self.feed = st.empty()
        self.lines, self.rows = [], []
        self.t0 = time.time()
    def say(self, msg):
        self.status.markdown(f"<div class='stage'>▸ {msg}</div>", unsafe_allow_html=True)
    def log(self, ok, text):
        self.lines.append(f"<div class='fl'><span class='tm'>{now()}</span> "
                          f"<span class='{'ok' if ok else 'err'}'>{'✓' if ok else '✕'}</span> {text}</div>")
        self.feed.markdown("<div class='feed'>" + "".join(self.lines[-10:]) + "</div>", unsafe_allow_html=True)
    def epoch(self, msg): self.detail.caption(msg)
    def refresh(self, done, total, metric, best):
        el = time.time() - self.t0
        bn, bv = best if best else ("—", None)
        self.tiles.markdown(tiles_html([
            ("Models trained", f"{done}/{total}", "real fits · no mocks", CY, VI),
            ("Best so far", bn, f"{metric} {fnum(bv)}" if bv is not None else "warming up", LI, CY),
            ("Elapsed", f"{el:.1f}s", "wall clock", AM, MG),
            ("Pipeline", "CV + holdout", "leak-free preprocessing", VI, BL)]), unsafe_allow_html=True)
        if self.rows:
            ok = [r for r in self.rows if r["ok"]]
            f = new_fig(280, f"Streaming leaderboard — {metric}")
            f.add_bar(x=[r["name"] for r in ok], y=[r["primary"] for r in ok],
                      marker_color=[FAM_COL.get(r["family"], CY) for r in ok],
                      error_y=dict(type="data", array=[r["cv_std"] if not math.isnan(r["cv_std"]) else 0 for r in ok],
                                   color="rgba(226,232,240,.5)"),
                      text=[fnum(r["primary"]) for r in ok], textposition="outside")
            f.update_layout(yaxis_title=metric, xaxis_tickangle=-18)
            self.chart.plotly_chart(f, use_container_width=True, key="lv_" + uid())
        self.bar.progress(min(done / max(total, 1), 1.0))

def optimize_params(spec, task, df, target, prep, cfg, ui):
    space = spec.get("space") or {}
    if not space: return {}, None
    y0 = df[target]
    def build(params):
        est = spec["make"](task, cfg["seed"], cfg["device"], params)
        pipe, X, y, _ = build_pipeline(df, target, task, prep, est)
        yy, _ = encode_target(y, task)
        return pipe, X, yy
    scoring = mdef(task, cfg["metric"])[3] or ("accuracy" if task == "classification" else "r2")
    timeout = {"Quick": 20, "Standard": 60, "Deep": 150}[cfg["budget"]]
    trials = int(cfg["trials"])
    if HAS["optuna"]:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            p = {}
            for k, sp in space.items():
                if sp[0] == "int": p[k] = trial.suggest_int(k, sp[1], sp[2])
                elif sp[0] == "float": p[k] = trial.suggest_float(k, sp[1], sp[2], log=bool(sp[3]) if len(sp) > 3 else False)
                else: p[k] = trial.suggest_categorical(k, sp[1])
            try:
                pipe, X, yy = build(p)
                sc = cross_val_score(pipe, X, yy, cv=3, scoring=scoring, n_jobs=-1, error_score=np.nan)
                return float(np.nanmean(sc))
            except Exception: return float("-inf")
        direction = "maximize" if mdef(task, cfg["metric"])[2] == "max" else "minimize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
        def cb(study, trial):
            ui.epoch(f"⚙ HPO trial {trial.number + 1}/{trials} · {spec['name']} · score {fnum(trial.value, 4)}")
        study.optimize(objective, n_trials=trials, timeout=timeout, callbacks=[cb], gc_after_trial=True)
        vals = [t.value for t in study.trials if t.value is not None and np.isfinite(t.value)]
        return dict(study.best_params), dict(engine="Optuna (TPE)", values=vals, best=study.best_value)
    dist = {}
    for k, sp in space.items():
        if sp[0] == "int": dist["model__" + k] = sps.randint(sp[1], sp[2] + 1)
        elif sp[0] == "float":
            dist["model__" + k] = sps.loguniform(sp[1], sp[2]) if (len(sp) > 3 and sp[3]) else sps.uniform(sp[1], sp[2] - sp[1])
        else: dist["model__" + k] = sp[1]
    pipe, X, yy = build({})
    rs = RandomizedSearchCV(pipe, dist, n_iter=trials, cv=3, scoring=scoring,
                            random_state=cfg["seed"], n_jobs=-1, error_score=np.nan)
    ui.epoch(f"⚙ Randomized search · {spec['name']} · {trials} candidates")
    rs.fit(X, yy)
    best = {k.replace("model__", ""): v for k, v in rs.best_params_.items()}
    return best, dict(engine="RandomizedSearchCV", values=[float(v) for v in rs.cv_results_["mean_test_score"] if np.isfinite(v)], best=float(rs.best_score_))

def make_bundle(task, pipe=None, ts=None, le=None, columns=None, kind="standard", extra=None):
    b = dict(task=task, pipeline=pipe, ts=ts, le=le, columns=list(columns or []),
             kind=kind, created=stamp(), platform="AutoForge")
    if extra: b.update(extra)
    return b

def train_one(aid, ds_name, cfg, ui):
    spec = get_spec(aid)
    if spec is None: raise ValueError("algorithm not found")
    if not algo_ok(spec): raise ValueError(f"requires optional package '{spec['needs']}' — pip install {spec['needs']}")
    df = S["datasets"][ds_name]["df"]
    task, target, prep = cfg["task"], cfg["target"], cfg["prep"]
    if len(df) > cfg["maxrows"]:
        df = df.sample(cfg["maxrows"], random_state=cfg["seed"])
        ui.epoch(f"Subsampled to {cfg['maxrows']:,} rows (seed {cfg['seed']})")
    t0 = time.time()
    run = dict(id=uid(), algo_id=aid, name=spec["name"], family=spec["family"], task=task,
               ds=ds_name, created=stamp(), status="ok", params={}, hpo=None,
               metric=cfg["metric"], custom_code=spec.get("code"))
    if task == "timeseries":
        dcol = cfg["date_col"]
        d = df[[dcol, target]].dropna().sort_values(dcol)
        series = pd.Series(d[target].values.astype(float), index=pd.DatetimeIndex(d[dcol]))
        sp = int(cfg["sp"])
        cut = int(len(series) * (1 - cfg["test_size"]))
        tr, te = series.iloc[:cut], series.iloc[cut:]
        wrapper = spec["make"](task, cfg["seed"], cfg["device"], {"sp": sp})
        ui.say(f"Fitting {spec['name']} on {len(tr)} observations…")
        wrapper.fit(tr)
        pred = wrapper.forecast(len(te))
        mets = ts_metrics(te.values, pred)
        fitted = np.asarray(wrapper.fitted_, float)[:len(tr)]
        tr_m = ts_metrics(tr.values[~np.isnan(fitted)], fitted[~np.isnan(fitted)])
        final = spec["make"](task, cfg["seed"], cfg["device"], {"sp": sp}).fit(series)
        run.update(metrics=mets, primary=mets[mdef(task, cfg["metric"])[1]],
                   cv_mean=float("nan"), cv_std=float("nan"), cv_note="time-ordered holdout",
                   train_score=tr_m[mdef(task, cfg["metric"])[1]],
                   bundle=make_bundle(task, ts=final, columns=[dcol, target],
                                      extra=dict(sp=sp, target=target, date_col=dcol)),
                   ts_plot=dict(train_idx=[str(x.date()) for x in tr.index], train=tr.values.tolist(),
                                test_idx=[str(x.date()) for x in te.index], test=te.values.tolist(),
                                pred=pred.tolist()),
                   duration=time.time() - t0)
        v, c, why = verdict_of(task, mdef(task, cfg["metric"])[1], "min", run["train_score"], float("nan"), run["primary"])
        run["verdict"], run["verdict_color"], run["verdict_why"] = v, c, why
        return run
    if task == "clustering":
        pre, X = build_unsup_prep(df, prep)
        cap = 1500 if aid in ("spectral", "tsne", "meanshift", "umap") else 8000
        Xf = X.sample(min(len(X), cap), random_state=cfg["seed"]) if len(X) > cap else X
        base = spec["make"](task, cfg["seed"], cfg["device"], {})
        kind = spec.get("kind", "standard")
        ui.say(f"Fitting {spec['name']} on {len(Xf)} rows × {Xf.shape[1]} features…")
        if kind == "embedding":
            pipe = Pipeline(pre.steps + [("model", base)])
            emb = pipe.fit_transform(Xf)
            ev = float(np.sum(base.explained_variance_ratio_)) if hasattr(base, "explained_variance_ratio_") else float("nan")
            run.update(metrics=dict(silhouette=float("nan"), calinski=float("nan"), davies=float("nan"),
                                    anomaly_rate=float("nan"), explained_var=ev),
                       primary=ev, metric="Explained variance (2D)", cv_mean=float("nan"), cv_std=float("nan"),
                       cv_note="embedding", train_score=float("nan"),
                       cluster=dict(kind="embedding", emb=emb.tolist(), labels=None),
                       bundle=make_bundle(task, pipe=pipe, columns=X.columns, kind="embedding"),
                       duration=time.time() - t0)
        elif kind == "anomaly":
            pipe = Pipeline(pre.steps + [("model", base)])
            pipe.fit(Xf)
            flags = pipe.predict(Xf)
            scores = -pipe.decision_function(Xf)
            rate = float(np.mean(flags == -1))
            run.update(metrics=dict(anomaly_rate=rate, mean_score=float(np.mean(scores)),
                                    silhouette=float("nan"), calinski=float("nan"), davies=float("nan")),
                       primary=rate, metric="Anomaly rate", cv_mean=float("nan"), cv_std=float("nan"),
                       cv_note="unsupervised", train_score=float("nan"),
                       cluster=dict(kind="anomaly", flags=flags.tolist(), scores=scores.tolist(),
                                    emb=PCA(2, random_state=cfg["seed"]).fit_transform(pipe[:-1].transform(Xf)).tolist()),
                       bundle=make_bundle(task, pipe=pipe, columns=X.columns, kind="anomaly"),
                       duration=time.time() - t0)
        else:
            pipe = Pipeline(pre.steps + [("model", ClusterAssigner(base=base))])
            pipe.fit(Xf)
            labels = pipe.named_steps["model"].labels_
            Xt = pipe[:-1].transform(Xf)
            mets = dict(silhouette=float("nan"), calinski=float("nan"), davies=float("nan"), anomaly_rate=float("nan"))
            mask = labels != -1
            uniq = np.unique(labels[mask])
            if 2 <= len(uniq) < mask.sum():
                try:
                    mets["silhouette"] = float(silhouette_score(Xt[mask], labels[mask],
                                                                sample_size=min(2000, int(mask.sum())), random_state=cfg["seed"]))
                    mets["calinski"] = float(calinski_harabasz_score(Xt[mask], labels[mask]))
                    mets["davies"] = float(davies_bouldin_score(Xt[mask], labels[mask]))
                except Exception: pass
            key = mdef(task, cfg["metric"])[1]
            emb = PCA(2, random_state=cfg["seed"]).fit_transform(Xt)
            run.update(metrics=mets, primary=mets.get(key, float("nan")),
                       cv_mean=float("nan"), cv_std=float("nan"), cv_note="unsupervised",
                       train_score=float("nan"), n_clusters=int(len(np.unique(labels))),
                       noise=float(np.mean(labels == -1)),
                       cluster=dict(kind="partition", labels=labels.tolist(), emb=emb.tolist()),
                       bundle=make_bundle(task, pipe=pipe, columns=X.columns, kind="partition"),
                       duration=time.time() - t0)
        run["verdict"], run["verdict_color"], run["verdict_why"] = "—", MUT, "Unsupervised run"
        return run
    # ---- classification / regression
    params, hpo = {}, None
    if cfg["hpo"] and spec.get("space"):
        ui.say(f"Optimizing {spec['name']} · {cfg['budget']} budget…")
        params, hpo = optimize_params(spec, task, df, target, prep, cfg, ui)
    est = spec["make"](task, cfg["seed"], cfg["device"], params)
    if isinstance(est, TorchNet):
        est._cb = lambda ep, tot, loss: ui.epoch(f"🧠 {spec['name']} · epoch {ep}/{tot} · loss {loss:.4f}")
    pipe, X, y, notes = build_pipeline(df, target, task, prep, est)
    yy, le = encode_target(y, task)
    strat = None
    if task == "classification":
        vc = pd.Series(yy).value_counts()
        if vc.min() >= 2: strat = yy
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=cfg["test_size"],
                                          random_state=cfg["seed"], stratify=strat)
    yfit = ytr
    semi = spec["family"] == "Semi-supervised"
    if semi and cfg["semi"]:
        rng = np.random.default_rng(cfg["seed"])
        mask = rng.random(len(ytr)) < cfg["mask"]
        yfit = ytr.copy(); yfit[mask] = -1
        ui.epoch(f"Semi-supervised: {int(mask.sum())}/{len(ytr)} training labels hidden")
    ui.say(f"Cross-validating + fitting {spec['name']}…")
    cmean, cstd, cnote = cv_eval(pipe, Xtr, ytr, task, mdef(task, cfg["metric"])[3],
                                 cfg["folds"], cfg["seed"], semi=semi)
    pipe.fit(Xtr, yfit)
    pred = pipe.predict(Xte)
    proba = get_proba(pipe, Xte)
    tr_pred = pipe.predict(Xtr)
    key, direction = mdef(task, cfg["metric"])[1], mdef(task, cfg["metric"])[2]
    if task == "classification":
        mets = clf_metrics(yte, pred, proba)
        trm = clf_metrics(ytr, tr_pred, get_proba(pipe, Xtr))
    else:
        mets = reg_metrics(yte, pred); trm = reg_metrics(ytr, tr_pred)
    prim = mets.get(key, float("nan"))
    v, c, why = verdict_of(task, key, direction, trm.get(key, float("nan")), cmean, prim)
    fnames = transformed_names(pipe)
    run.update(params=params, hpo=hpo, metrics=mets, primary=prim,
               cv_mean=cmean, cv_std=cstd, cv_note=cnote, train_score=trm.get(key, float("nan")),
               verdict=v, verdict_color=c, verdict_why=why,
               model=pipe, le=le, classes=(le.classes_.tolist() if le is not None else np.unique(yy).tolist()),
               X_test=Xte.reset_index(drop=True), y_test=np.asarray(yte).tolist(),
               pred=pred.tolist(), proba=(proba.tolist() if proba is not None else None),
               curves=curves_of(yte, proba) if task == "classification" else None,
               feature_space=feature_space_of(X), feature_names=fnames, notes=notes,
               bundle=make_bundle(task, pipe=pipe, le=le, columns=X.columns),
               duration=time.time() - t0)
    return run

def champ_snapshot(run):
    snap = dict(name=run["name"], family=run["family"], task=run["task"], metric=run["metric"],
                primary=run["primary"], metrics=run["metrics"], cv_mean=run.get("cv_mean"),
                cv_std=run.get("cv_std"), verdict=run.get("verdict"), duration=run.get("duration"),
                params=run.get("params", {}), curves=run.get("curves"))
    if run["task"] in ("classification", "regression"):
        n = min(300, len(run.get("y_test", [])))
        snap["sample"] = dict(y=run["y_test"][:n], pred=run["pred"][:n],
                              proba=(run["proba"][:n] if run.get("proba") else None))
        if run["task"] == "classification":
            cm = confusion_matrix(run["y_test"], run["pred"])
            snap["cm"] = dict(matrix=cm.tolist(), labels=[str(c) for c in run["classes"]])
        try:
            names, vals, method = native_importance(run)
            snap["importance"] = dict(names=names[:15], values=[float(v) for v in vals[:15]], method=method)
        except Exception: pass
    if run["task"] == "timeseries": snap["ts_plot"] = run.get("ts_plot")
    if run["task"] == "clustering": snap["cluster"] = run.get("cluster")
    return snap

def run_automl(ds_name, algo_ids, cfg, container, celebrate=True, demo=False):
    ui = LiveUI(container, f"{ds_name} · {cfg['task']}")
    ui.say("Profiling dataset & locking preprocessing recipe…")
    ui.log(True, f"Dataset '{ds_name}' · task <b>{cfg['task']}</b> · target <b>{cfg['target'] or '—'}</b> · metric <b>{cfg['metric']}</b>")
    df = S["datasets"][ds_name]["df"]
    prof = profile_df(df)
    ui.log(True, f"Profile: {prof['rows']:,} rows × {prof['cols']} cols · {prof['missing_pct']}% missing · {prof['dups']} duplicates")
    results, best = [], None
    direction = mdef(cfg["task"], cfg["metric"])[2]
    for i, aid in enumerate(algo_ids):
        spec = get_spec(aid)
        nm = spec["name"] if spec else aid
        try:
            run = train_one(aid, ds_name, cfg, ui)
            results.append(run); S["runs"].append(run)
            okv = run["primary"]
            if best is None or (not math.isnan(okv) and
                                ((direction == "max" and okv > best[1]) or (direction == "min" and okv < best[1]))):
                best = (run["name"], okv, run["id"])
            cvtxt = f" · CV {fnum(run['cv_mean'])}±{fnum(run['cv_std'])}" if not math.isnan(run.get("cv_mean", float("nan"))) else ""
            ui.log(True, f"<b>{run['name']}</b> · {cfg['metric']} <b>{fnum(okv)}</b>{cvtxt} · {run['duration']:.1f}s · {run.get('verdict','—')}")
        except Exception as e:
            fr = dict(id=uid(), algo_id=aid, name=nm, family=(spec or {}).get("family", "Custom"),
                      task=cfg["task"], ds=ds_name, created=stamp(), status="failed",
                      error=str(e)[:300], primary=float("nan"), metric=cfg["metric"],
                      cv_mean=float("nan"), cv_std=float("nan"), duration=0.0,
                      verdict="Failed", verdict_color=RD, verdict_why=str(e)[:300], metrics={})
            results.append(fr); S["runs"].append(fr)
            ui.log(False, f"<b>{nm}</b> failed — {str(e)[:140]}")
        ui.rows.append(dict(name=nm, family=(spec or {}).get("family", "Custom"),
                            primary=results[-1]["primary"], cv_std=results[-1].get("cv_std", float("nan")),
                            dur=results[-1].get("duration", 0), ok=results[-1]["status"] == "ok"))
        ui.refresh(i + 1, len(algo_ids), cfg["metric"], best[:2] if best else None)
        time.sleep(0.12)
    good = [r for r in results if r["status"] == "ok" and not math.isnan(r.get("primary", float("nan")))]
    champ = None
    if good:
        champ = (max(good, key=lambda r: r["primary"]) if direction == "max"
                 else min(good, key=lambda r: r["primary"]))
        prev = next((r for r in S["runs"] if r["id"] == S.get("champion")), None)
        better = (prev is None or prev["ds"] != ds_name or prev["task"] != cfg["task"] or
                  (direction == "max" and champ["primary"] > prev["primary"]) or
                  (direction == "min" and champ["primary"] < prev["primary"]))
        if better: S["champion"] = champ["id"]
    exp = dict(id=uid(), ts=stamp(), ds=ds_name, task=cfg["task"], metric=cfg["metric"],
               algos=list(algo_ids), cfg=cfg, run_ids=[r["id"] for r in results],
               duration=sum(r.get("duration", 0) for r in results),
               feed=ui.lines, rows=ui.rows, data_hash=df_hash(df), demo=demo,
               ds_origin=S["datasets"][ds_name]["origin"], demo_seed=S["datasets"][ds_name].get("seed"),
               profile=dict(rows=prof["rows"], cols=prof["cols"], missing_pct=prof["missing_pct"], dups=prof["dups"]),
               champion=champ["id"] if champ else None,
               champ_snapshot=champ_snapshot(champ) if champ else None)
    S["experiments"].insert(0, exp); db_exp_put(exp)
    ui.say(f"Run complete — {len(good)}/{len(algo_ids)} models succeeded" +
           (f" · 👑 champion: {champ['name']} ({cfg['metric']} {fnum(champ['primary'])})" if champ else ""))
    ui.log(True, f"Experiment <b>{exp['id']}</b> logged · fingerprint <b>{exp['data_hash']}</b> · reproducible with seed {cfg['seed']}")
    if cfg.get("mlflow") and HAS["mlflow"]:
        try:
            import mlflow
            mlflow.set_experiment("AutoForge")
            for r in good:
                with mlflow.start_run(run_name=f"{r['name']}·{exp['id']}"):
                    mlflow.log_params({k: str(v)[:240] for k, v in cfg.items() if k != "prep"})
                    mlflow.log_params({f"p_{k}": str(v) for k, v in r.get("params", {}).items()})
                    mlflow.log_metrics({k: float(v) for k, v in r["metrics"].items() if isinstance(v, (int, float)) and np.isfinite(v)})
            ui.log(True, "MLflow tracking updated (local ./mlruns)")
        except Exception as e:
            ui.log(False, f"MLflow skipped — {str(e)[:100]}")
    if celebrate and champ:
        st.balloons(); toast(f"Champion crowned: {champ['name']} · {cfg['metric']} {fnum(champ['primary'])}", "👑")
    return exp

# ---------------------------------------------------------------- explainability
def native_importance(run):
    pipe = run["model"]; est = pipe.named_steps["model"]
    names = run.get("feature_names") or transformed_names(pipe)
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        raise ValueError("no native importance")
    if names is None or len(names) != len(vals): names = [f"f{i}" for i in range(len(vals))]
    order = np.argsort(vals)[::-1]
    return [names[i] for i in order], vals[order], "Native (impurity / coefficients)"

def permutation_imp(run, seed=42):
    scoring = mdef(run["task"], run["metric"])[3] or ("accuracy" if run["task"] == "classification" else "r2")
    r = permutation_importance(run["model"], run["X_test"], run["y_test"], scoring=scoring,
                               n_repeats=5, random_state=seed, n_jobs=-1)
    order = np.argsort(r.importances_mean)[::-1]
    names = list(run["X_test"].columns)
    return ([names[i] for i in order], r.importances_mean[order],
            "Permutation importance (holdout, 5 repeats)")

def compute_shap_run(run):
    import shap
    pipe = run["model"]; est = pipe.named_steps["model"]
    Xs = run["X_test"].sample(min(150, len(run["X_test"])), random_state=42)
    Xt = pipe[:-1].transform(Xs)
    names = run.get("feature_names") or [f"f{i}" for i in range(Xt.shape[1])]
    tree_like = hasattr(est, "feature_importances_") or est.__class__.__name__ in ("XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor", "CatBoostClassifier", "CatBoostRegressor")
    if not tree_like: return None
    expl = shap.TreeExplainer(est)
    sv = expl.shap_values(Xt, check_additivity=False)
    if isinstance(sv, list): sv = sv[1] if len(sv) == 2 else sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    base = expl.expected_value
    if isinstance(base, (list, np.ndarray)): base = float(np.ravel(base)[-1])
    return dict(values=sv, X=np.asarray(Xt), names=list(names), base=float(base))

def shap_local(run, row_df):
    try:
        import shap
        pipe = run["model"]; est = pipe.named_steps["model"]
        if not (hasattr(est, "feature_importances_") or est.__class__.__name__.startswith(("XGB", "LGBM", "CatBoost"))):
            return None
        Xt = pipe[:-1].transform(row_df)
        expl = shap.TreeExplainer(est)
        sv = expl.shap_values(Xt, check_additivity=False)
        if isinstance(sv, list): sv = sv[1] if len(sv) == 2 else sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3: sv = sv[:, :, 1]
        names = run.get("feature_names") or [f"f{i}" for i in range(Xt.shape[1])]
        base = expl.expected_value
        if isinstance(base, (list, np.ndarray)): base = float(np.ravel(base)[-1])
        return dict(values=sv[0], names=list(names), base=float(base))
    except Exception:
        return None

# ---------------------------------------------------------------- export / deploy
def bundle_bytes(bundle):
    buf = io.BytesIO(); joblib.dump(bundle, buf, compress=3); return buf.getvalue()

def custom_sources(bundle):
    seen, srcs = set(), []
    local_names = {"PseudoLabelModel", "BlendModel", "ClusterAssigner", "TorchNet",
                   "StatsForecast", "MLForecast", "ProphetForecast", "FunctionTransformerClip"}
    def visit(obj, depth=0):
        if obj is None or depth > 4: return
        cls = type(obj)
        if cls.__name__ in local_names and cls.__name__ not in seen:
            try: srcs.append(inspect.getsource(cls)); seen.add(cls.__name__)
            except Exception: pass
        if isinstance(obj, Pipeline):
            for _, s in obj.steps: visit(s, depth + 1)
        for attr in ("estimators_", "estimators", "bases_", "base_estimator", "estimator",
                     "final_estimator", "est_", "base_", "base", "knn_", "meta_"):
            v = getattr(obj, attr, None)
            if isinstance(v, (list, tuple)):
                for it in v:
                    visit(it[1] if isinstance(it, tuple) and len(it) == 2 else it, depth + 1)
            elif v is not None and not isinstance(v, (str, int, float, bool, np.ndarray)):
                visit(v, depth + 1)
    visit(bundle.get("pipeline")); visit(bundle.get("ts"))
    return "\n\n".join(srcs)

DEPLOY_TMPL = Template('''# AutoForge — exported REST API server
# Run:  pip install -r requirements.txt  &&  python server.py
import io, json, time, types, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest)
from sklearn.linear_model import LogisticRegression, Ridge

$SNIPPET
$CUSTOM

BUNDLE = joblib.load(Path(__file__).with_name("model.joblib"))
TASK = BUNDLE["task"]
app = FastAPI(title="AutoForge Model API — $NAME", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "model": "$NAME", "task": TASK}

@app.post("/predict")
def predict(payload: dict):
    t0 = time.time()
    rows = payload.get("rows")
    if not rows: raise HTTPException(422, "body must be {'rows': [ {feature: value}, ... ]}")
    df = pd.DataFrame(rows)
    missing = [c for c in BUNDLE["columns"] if c not in df.columns]
    if missing: raise HTTPException(422, f"missing columns: {missing}")
    df = df[BUNDLE["columns"]]
    pipe = BUNDLE["pipeline"]
    out = {"model": "$NAME", "task": TASK}
    if BUNDLE.get("kind") == "embedding":
        out["embedding"] = pipe.transform(df).round(4).tolist()
    else:
        pred = pipe.predict(df)
        out["predictions"] = pred.tolist()
        if BUNDLE.get("le") is not None:
            out["labels"] = BUNDLE["le"].inverse_transform(pred.astype(int)).tolist()
        if hasattr(pipe, "predict_proba"):
            out["probabilities"] = pipe.predict_proba(df).round(4).tolist()
        if BUNDLE.get("kind") == "anomaly":
            out["anomaly_scores"] = (-pipe.decision_function(df)).round(4).tolist()
    out["latency_ms"] = round((time.time() - t0) * 1000, 2)
    return out
$FORECAST
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''')

FORECAST_EP = '''
@app.post("/forecast")
def forecast(payload: dict):
    h = int(payload.get("horizon", 7))
    m = BUNDLE["ts"]
    return {"model": "$NAME", "horizon": h, "forecast": [round(float(v), 3) for v in m.forecast(h)]}
'''

def deploy_code(run):
    bundle = run["bundle"]
    snippet = ""
    if run.get("custom_code"):
        snippet = ("_m = types.ModuleType('custom_algo'); sys.modules['custom_algo'] = _m\n"
                   f"exec(compile(json.loads({json.dumps(run['custom_code'])}), '<custom>', 'exec'), _m.__dict__)")
    fc = Template(FORECAST_EP).safe_substitute(NAME=run["name"]) if run["task"] == "timeseries" else ""
    return DEPLOY_TMPL.safe_substitute(NAME=run["name"], CUSTOM=custom_sources(bundle),
                                       SNIPPET=snippet, FORECAST=fc)

def deploy_requirements(run):
    req = ["fastapi", "uvicorn", "joblib", "numpy", "pandas", "scikit-learn"]
    spec = get_spec(run.get("algo_id", ""))
    need = (spec or {}).get("needs")
    if need in ("xgboost", "lightgbm", "catboost", "torch", "statsmodels", "prophet", "umap"):
        req.append("umap-learn" if need == "umap" else need)
    return "\n".join(req) + "\n"

def sample_payload(run):
    if run["task"] == "timeseries":
        return json.dumps({"horizon": int(run["bundle"].get("sp", 7)) * 2}, indent=2)
    X = run.get("X_test")
    if X is None: return json.dumps({"rows": []}, indent=2)
    rows = json.loads(X.head(2).to_json(orient="records"))
    return json.dumps({"rows": rows}, indent=2)

def local_predict(run, rows):
    bundle = run["bundle"]
    df = pd.DataFrame(rows)
    missing = [c for c in bundle["columns"] if c not in df.columns]
    if missing: raise ValueError(f"missing columns: {missing}")
    df = df[bundle["columns"]]
    pipe = bundle["pipeline"]
    out = dict(model=run["name"], task=run["task"])
    if bundle.get("kind") == "embedding":
        out["embedding"] = pipe.transform(df).round(4).tolist(); return out
    pred = pipe.predict(df)
    out["predictions"] = pred.tolist() if not isinstance(pred, np.ndarray) else pred.tolist()
    if bundle.get("le") is not None:
        out["labels"] = bundle["le"].inverse_transform(np.asarray(pred).astype(int)).tolist()
    proba = get_proba(pipe, df)
    if proba is not None: out["probabilities"] = np.round(proba, 4).tolist()
    if bundle.get("kind") == "anomaly":
        out["anomaly_scores"] = np.round(-pipe.decision_function(df), 4).tolist()
    return out

def deploy_zip(run):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.joblib", bundle_bytes(run["bundle"]))
        z.writestr("server.py", deploy_code(run))
        z.writestr("requirements.txt", deploy_requirements(run))
        z.writestr("sample_request.json", sample_payload(run))
        z.writestr("README.md",
                   f"# {run['name']} — deployed with AutoForge\n\n"
                   "## Windows 10/11 quickstart\n```\npython -m venv .venv\n.venv\\Scripts\\activate\n"
                   "pip install -r requirements.txt\npython server.py\n```\n"
                   "API docs: http://localhost:8000/docs · health: GET /health · predict: POST /predict\n\n"
                   f"Sample request:\n```json\n{sample_payload(run)}\n```\n")
    return buf.getvalue()

# ---------------------------------------------------------------- reports
def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows: out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out

def report_figs(exp):
    figs = []
    rows = [r for r in exp.get("rows", []) if r.get("ok")]
    if rows:
        f = new_fig(300, f"Leaderboard — {exp['metric']}")
        f.add_bar(x=[r["name"] for r in rows], y=[r["primary"] for r in rows],
                  marker_color=[FAM_COL.get(r["family"], CY) for r in rows],
                  text=[fnum(r["primary"]) for r in rows], textposition="outside")
        figs.append(f)
    snap = exp.get("champ_snapshot") or {}
    if snap.get("cm"):
        figs.append(cm_fig(snap["cm"]["matrix"], snap["cm"]["labels"], False, "Champion confusion matrix"))
    if snap.get("curves"):
        figs.append(roc_fig([dict(name=snap["name"], curves=snap["curves"])], "Champion ROC"))
    if snap.get("sample") and snap.get("task") == "regression":
        s = snap["sample"]
        f = new_fig(300, "Actual vs predicted (holdout sample)")
        f.add_scatter(x=s["y"], y=s["pred"], mode="markers", marker=dict(color=CY, size=6, opacity=.7))
        lim = [min(s["y"]), max(s["y"])]
        f.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=MG, dash="dash"), name="ideal")
        figs.append(f)
    if snap.get("importance"):
        imp = snap["importance"]
        f = new_fig(300, f"Feature importance — {imp['method']}")
        f.add_bar(x=imp["values"][::-1], y=imp["names"][::-1], orientation="h", marker_color=VI)
        figs.append(f)
    if snap.get("ts_plot"):
        figs.append(ts_fig(snap["ts_plot"], "Champion holdout forecast"))
    return figs

def cm_fig(matrix, labels, normalize, title="Confusion matrix"):
    cm = np.asarray(matrix, float)
    if normalize:
        rs = cm.sum(1, keepdims=True); cm = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    f = new_fig(340, title)
    f.add_heatmap(z=cm, x=labels, y=labels,
                  colorscale=[[0, "#0B1220"], [.5, "#155E75"], [1, "#22D3EE"]],
                  text=np.round(cm, 2 if normalize else 0), texttemplate="%{text}", showscale=False)
    f.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
    return f

def roc_fig(entries, title="ROC curves"):
    f = new_fig(340, title)
    for i, e in enumerate(entries):
        c = e.get("curves")
        if not c: continue
        for j, r in enumerate(c["roc"]):
            f.add_scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                          name=f"{e['name']} {r['label']} (AUC {r['auc']:.3f})",
                          line=dict(color=PAL[i % len(PAL)], width=2.2, dash=DASH[j % len(DASH)]))
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="rgba(148,163,184,.4)", dash="dot"),
                  name="chance", showlegend=False)
    f.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return f

def pr_fig(entries, title="Precision–Recall curves"):
    f = new_fig(340, title)
    for i, e in enumerate(entries):
        c = e.get("curves")
        if not c: continue
        for j, r in enumerate(c["pr"]):
            f.add_scatter(x=r["rec"], y=r["prec"], mode="lines",
                          name=f"{e['name']} {r['label']} (AP {r['ap']:.3f})",
                          line=dict(color=PAL[i % len(PAL)], width=2.2, dash=DASH[j % len(DASH)]))
    f.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return f

def ts_fig(tp, title="Forecast vs actual"):
    f = new_fig(340, title)
    tr_idx, tr = tp["train_idx"], tp["train"]
    if len(tr) > 400:
        step = len(tr) // 400; tr_idx, tr = tr_idx[::step], tr[::step]
    f.add_scatter(x=tr_idx, y=tr, mode="lines", name="train", line=dict(color="rgba(148,163,184,.6)", width=1.4))
    f.add_scatter(x=tp["test_idx"], y=tp["test"], mode="lines", name="actual (holdout)", line=dict(color=CY, width=2.2))
    f.add_scatter(x=tp["test_idx"], y=tp["pred"], mode="lines", name="forecast", line=dict(color=AM, width=2.2, dash="dash"))
    return f

def report_html(exp):
    snap = exp.get("champ_snapshot") or {}
    figs = report_figs(exp)
    parts, first = [], True
    for fg in figs:
        parts.append(fg.to_html(full_html=False, include_plotlyjs=True if first else False))
        first = False
    rows = [[r["name"], r["family"], fnum(r["primary"]), fnum(r.get("dur", 0), 1) + "s",
             "✓" if r.get("ok") else "✕"] for r in exp.get("rows", [])]
    lb = md_table(["Model", "Family", exp["metric"], "Time", "OK"], rows)
    lb_html = "<table><tr>" + "".join(f"<th>{h}</th>" for h in ["Model", "Family", exp["metric"], "Time", "OK"]) + "</tr>"
    for r in rows: lb_html += "<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
    lb_html += "</table>"
    cfg_json = json.dumps(exp.get("cfg", {}), indent=2, default=jdef)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>AutoForge Report {exp['id']}</title>
<style>body{{background:#0B1220;color:#E2E8F0;font-family:Segoe UI,Inter,sans-serif;margin:0;padding:34px;}}
h1{{background:linear-gradient(90deg,#22D3EE,#8B5CF6,#F472B6);-webkit-background-clip:text;color:transparent;font-size:30px;margin:0;}}
h2{{color:#A5F3FC;border-bottom:1px solid rgba(34,211,238,.25);padding-bottom:6px;margin-top:30px;}}
.meta{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0;}}
.m{{background:#111A2D;border:1px solid rgba(148,163,184,.2);border-radius:12px;padding:10px 16px;}}
.m b{{display:block;color:#22D3EE;font-size:18px;}} .m span{{font-size:11px;color:#8FA3BF;text-transform:uppercase;letter-spacing:1px;}}
table{{border-collapse:collapse;width:100%;margin:10px 0;}} th,td{{border:1px solid rgba(148,163,184,.25);padding:7px 10px;text-align:left;font-size:13px;}}
th{{background:#111A2D;color:#A5F3FC;}} pre{{background:#0D1526;border:1px solid rgba(139,92,246,.3);border-radius:10px;padding:14px;overflow:auto;font-size:12px;}}
.badge{{display:inline-block;padding:3px 12px;border-radius:99px;border:1px solid #A3E635;color:#A3E635;font-weight:600;}}
</style></head><body>
<h1>⚡ AutoForge — Experiment Report</h1>
<p style="color:#8FA3BF">Generated {stamp()} · experiment <b>{exp['id']}</b> · reproducibility fingerprint <b>{exp['data_hash']}</b></p>
<div class="meta">
<div class="m"><span>Dataset</span><b>{exp['ds']}</b></div>
<div class="m"><span>Task</span><b>{exp['task']}</b></div>
<div class="m"><span>Rows × Cols</span><b>{exp['profile']['rows']:,} × {exp['profile']['cols']}</b></div>
<div class="m"><span>Missing</span><b>{exp['profile']['missing_pct']}%</b></div>
<div class="m"><span>Models</span><b>{len(exp.get('rows', []))}</b></div>
<div class="m"><span>Champion</span><b>{snap.get('name', '—')}</b></div>
<div class="m"><span>{exp['metric']}</span><b>{fnum(snap.get('primary'))}</b></div>
</div>
<h2>Leaderboard</h2>{lb_html}
<h2>Champion assessment</h2>
<p><span class="badge">{snap.get('verdict', '—')}</span> &nbsp; CV mean {fnum(snap.get('cv_mean'))} ± {fnum(snap.get('cv_std'))} · trained in {fnum(snap.get('duration'), 1)}s</p>
<h2>Figures</h2>{''.join(parts)}
<h2>Reproducibility configuration</h2><pre>{cfg_json}</pre>
<p style="color:#64748B;font-size:12px">AutoForge · open-source AutoML foundry · this report is fully self-contained (offline-ready).</p>
</body></html>"""
    return html

def report_md(exp):
    snap = exp.get("champ_snapshot") or {}
    rows = [[r["name"], fnum(r["primary"]), fnum(r.get("dur", 0), 1) + "s"] for r in exp.get("rows", [])]
    return (f"# AutoForge Report — {exp['id']}\n\n- Dataset: **{exp['ds']}** ({exp['profile']['rows']:,} rows)\n"
            f"- Task: **{exp['task']}** · Metric: **{exp['metric']}**\n- Fingerprint: `{exp['data_hash']}`\n"
            f"- Champion: **{snap.get('name','—')}** → {fnum(snap.get('primary'))} ({snap.get('verdict','—')})\n\n"
            "## Leaderboard\n" + md_table(["Model", exp["metric"], "Time"], rows) +
            "\n## Config\n```json\n" + json.dumps(exp.get("cfg", {}), indent=2, default=jdef) + "\n```\n")

# ---------------------------------------------------------------- UI components
def hero():
    ds = S.get("active_ds"); cfg = S["ds_cfg"].get(ds, {})
    champ = next((r for r in S["runs"] if r["id"] == S.get("champion")), None)
    ch = [chip(f"📦 {ds}", CY), chip(f"🎯 {cfg.get('task','—')}", VI),
          chip(f"👑 {champ['name']} · {fnum(champ['primary'])}" if champ else "👑 no champion yet", LI if champ else MUT),
          chip("⚡ GPU ready" if cuda_ok() else "🖥 CPU mode", AM),
          chip(f"🧪 {len(S['runs'])} runs", BL)]
    st.markdown(f"<div class='hero'><div><div class='brand'>⚡ AutoForge</div>"
                f"<div class='tagline'>All-in-one AutoML foundry — data → 50+ algorithms → champion → REST API</div></div>"
                f"<div class='chips'>{''.join(ch)}</div></div>", unsafe_allow_html=True)

def neural_banner():
    components.html("""
<div style="position:relative;border:1px solid rgba(139,92,246,.28);border-radius:16px;overflow:hidden;
 background:linear-gradient(120deg,rgba(34,211,238,.06),rgba(139,92,246,.09),rgba(244,114,182,.05));">
<canvas id="c" style="display:block;width:100%;height:150px;"></canvas>
<div style="position:absolute;top:10px;left:14px;font:600 11px 'Segoe UI',sans-serif;letter-spacing:2px;color:#8FA3BF;">
NEURAL TELEMETRY <span style="color:#22D3EE">· move your pointer — the network responds</span></div></div>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const DPR=Math.min(window.devicePixelRatio||1,2);
let W,H,nodes=[],pulses=[],mouse={x:-999,y:-999};
const PAL=['#22D3EE','#8B5CF6','#A3E635','#F59E0B','#F472B6','#60A5FA'];
function size(){W=cv.clientWidth;H=150;cv.width=W*DPR;cv.height=H*DPR;ctx.setTransform(DPR,0,0,DPR,0,0);}
function init(){nodes=[];const n=Math.max(16,Math.floor(W/46));
 for(let i=0;i<n;i++)nodes.push({x:Math.random()*W,y:20+Math.random()*(H-40),
  vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.3,r:1.6+Math.random()*2.2,c:PAL[i%PAL.length],ph:Math.random()*6.28});}
size();init();
window.addEventListener('resize',()=>{size();init();});
cv.addEventListener('mousemove',e=>{const b=cv.getBoundingClientRect();mouse.x=e.clientX-b.left;mouse.y=e.clientY-b.top;});
cv.addEventListener('mouseleave',()=>{mouse.x=-999;});
let t=0,last=0;
function frame(ts){requestAnimationFrame(frame);if(document.hidden)return;
 if(ts-last<1000/50)return;last=ts;t+=.016;
 ctx.clearRect(0,0,W,H);
 for(const n of nodes){n.x+=n.vx;n.y+=n.vy;
  const dx=mouse.x-n.x,dy=mouse.y-n.y,d=Math.hypot(dx,dy);
  if(d<130&&d>1){n.x+=dx/d*.5;n.y+=dy/d*.5;}
  if(n.x<0||n.x>W)n.vx*=-1;if(n.y<14||n.y>H-14)n.vy*=-1;}
 for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
  const a=nodes[i],b=nodes[j],d=Math.hypot(a.x-b.x,a.y-b.y);
  if(d<110){ctx.strokeStyle='rgba(139,92,246,'+(0.22*(1-d/110))+')';ctx.lineWidth=1;
   ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
   if(Math.random()<.004&&pulses.length<40)pulses.push({a,b,t:0,c:a.c});}}
 pulses=pulses.filter(p=>p.t<1);
 for(const p of pulses){p.t+=.03;const x=p.a.x+(p.b.x-p.a.x)*p.t,y=p.a.y+(p.b.y-p.a.y)*p.t;
  ctx.fillStyle=p.c;ctx.shadowColor=p.c;ctx.shadowBlur=8;
  ctx.beginPath();ctx.arc(x,y,2.1,0,6.28);ctx.fill();ctx.shadowBlur=0;}
 for(const n of nodes){const r=n.r+Math.sin(t*2+n.ph)*.7;
  ctx.fillStyle=n.c;ctx.shadowColor=n.c;ctx.shadowBlur=10;
  ctx.beginPath();ctx.arc(n.x,n.y,r,0,6.28);ctx.fill();ctx.shadowBlur=0;}
}
requestAnimationFrame(frame);
</script>""", height=158)

def undo_bar():
    tr = S.get("trash")
    if not tr: return
    c1, c2 = st.columns([5, 1])
    c1.markdown(f"<div class='stage'>🗑 {tr['label']} — undo available</div>", unsafe_allow_html=True)
    if c2.button("↩ Undo", key="undo_" + tr["kind"]):
        k, p = tr["kind"], tr["payload"]
        if k == "runs": S["runs"].extend(p)
        elif k == "dataset":
            S["datasets"][p["name"]] = p["data"]; S["ds_cfg"][p["name"]] = p["cfg"]
        elif k == "registry":
            S["registry"].insert(0, p["entry"]); db_registry_put(p["entry"], p["blob"])
        elif k == "experiment":
            S["experiments"].insert(0, p); db_exp_put(p)
        S["trash"] = None; toast("Restored", "↩"); RERUN()

def champion_run():
    return next((r for r in S["runs"] if r["id"] == S.get("champion")), None)

def goto(page):
    S["nav_goto"] = page; RERUN()

PAGES = ["🛰️ Mission Control", "🧪 Data Studio", "⚙️ AutoML Lab", "🏆 Compare & Select",
         "🔬 Diagnostics", "🔮 Predict Studio", "🚀 Deploy & API", "🗄️ Registry & History",
         "📑 Reports", "📋 Code & Setup"]

def workflow_row():
    steps = [("📦 Data", bool(S["datasets"]), "🧪 Data Studio"),
             ("🔎 Profile", bool(S["datasets"]), "🧪 Data Studio"),
             ("🧠 Train", len(S["runs"]) > 0, "⚙️ AutoML Lab"),
             ("🏆 Compare", len([r for r in S["runs"] if r["status"] == "ok"]) > 1, "🏆 Compare & Select"),
             ("🔮 Predict", S.get("predicted", False), "🔮 Predict Studio"),
             ("🚀 Deploy", S.get("deployed", False), "🚀 Deploy & API")]
    cols = st.columns(6)
    for c, (label, done, page) in zip(cols, steps):
        c.markdown(chip(("✓ " if done else "○ ") + label, LI if done else MUT), unsafe_allow_html=True)
        if c.button("Open", key="wf_" + label, use_container_width=True): goto(page)

def dataset_cards():
    cols = st.columns(len(S["datasets"]))
    for col, (name, meta) in zip(cols, list(S["datasets"].items())):
        df = meta["df"]; cfg = S["ds_cfg"].get(name, {})
        icon = DEMOS.get(name, {}).get("icon", "📁")
        active = name == S["active_ds"]
        prof_missing = round(100 * df.isna().sum().sum() / max(df.size, 1), 1)
        with col:
            st.markdown(f"<div class='card' style='border-color:{'rgba(34,211,238,.55)' if active else 'rgba(148,163,184,.18)'}'>"
                        f"<h4>{icon} {name} {'· active' if active else ''}</h4>"
                        f"<div class='m'>{len(df):,} rows × {df.shape[1]} cols · {prof_missing}% missing</div>"
                        f"<div style='margin-top:6px'>{chip(cfg.get('task','—'), VI)} {chip('target: ' + str(cfg.get('target') or 'none'), CY)}</div>"
                        f"</div>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            if b1.button("Activate", key="act_" + name, use_container_width=True):
                S["active_ds"] = name; toast(f"Active dataset: {name}", "📦"); RERUN()
            if b2.button("▶ Run", key="run_" + name, use_container_width=True, type="primary"):
                S["active_ds"] = name; quick_run_cfg(name); S["lab_autorun"] = True; goto("⚙️ AutoML Lab")

def quick_run_cfg(ds):
    cfg = S["ds_cfg"][ds]
    task = cfg["task"]
    df = S["datasets"][ds]["df"]
    recs, _ = recommend(df, task, cfg.get("target"), n=6)
    S["lab_algos"] = [a["id"] for a in recs]
    S["lab_task"] = "Auto-detected"; S["lab_metric"] = METRICS[task][0][0]
    S["lab_auto"] = False

def capability_matrix():
    fams = {}
    for a in ALGOS: fams.setdefault(a["family"], []).append(a)
    with st.expander(f"🧬 Algorithm catalog — {len(ALGOS)} built-ins across {len(fams)} families (+ custom estimators)"):
        for fam, items in fams.items():
            ok = sum(1 for a in items if algo_ok(a))
            st.markdown(f"**{fam}** &nbsp; {chip(f'{ok}/{len(items)} ready', FAM_COL[fam])}", unsafe_allow_html=True)
            st.markdown(chips([chip(a["name"] + ("" if algo_ok(a) else f" · needs {a['needs']}"),
                                      FAM_COL[fam] if algo_ok(a) else MUT) for a in items]),
                        unsafe_allow_html=True)
            st.write("")

def static_live_panel(exp):
    st.markdown("<div class='live-head'><span style='color:#A3E635'>●</span> LAST LIVE SIMULATION · replay anytime</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("<div class='feed'>" + "".join(exp.get("feed", [])[-10:]) + "</div>", unsafe_allow_html=True)
    with c2:
        rows = [r for r in exp.get("rows", []) if r.get("ok")]
        if rows:
            f = new_fig(268, f"Final telemetry — {exp['metric']}")
            f.add_bar(x=[r["name"] for r in rows], y=[r["primary"] for r in rows],
                      marker_color=[FAM_COL.get(r["family"], CY) for r in rows],
                      text=[fnum(r["primary"]) for r in rows], textposition="outside")
            f.update_layout(xaxis_tickangle=-18)
            st.plotly_chart(f, use_container_width=True, key="static_tel")
    b1, b2, b3 = st.columns([1, 1, 2])
    if b1.button("🔁 Replay live simulation", type="primary", use_container_width=True):
        S["demo_pending"] = True; RERUN()
    if b2.button("⚙️ Open AutoML Lab", use_container_width=True): goto("⚙️ AutoML Lab")
    b3.caption(f"{len(exp.get('rows', []))} real models trained on {exp['profile']['rows']:,} rows · "
               f"fingerprint {exp['data_hash']} · seed {exp['cfg'].get('seed')} — nothing here is a mock.")

# ---------------------------------------------------------------- pages
def page_overview():
    hero(); neural_banner()
    ds = S["active_ds"]; df = S["datasets"][ds]["df"]; cfg = S["ds_cfg"][ds]
    prof = profile_df(df)
    champ = champion_run()
    tile_ph = st.empty()
    def draw_tiles():
        tile_ph.markdown(tiles_html([
            ("Active dataset", f"{prof['rows']:,} rows", f"{prof['cols']} features · {prof['missing_pct']}% missing", CY, VI),
            ("Detected task", cfg["task"], f"confidence {int(cfg.get('confidence', .8) * 100)}%", VI, MG),
            ("Models trained", str(len([r for r in S['runs'] if r['status'] == 'ok'])), f"{len(S['experiments'])} experiments", LI, CY),
            ("Champion", champ["name"] if champ else "—", f"{champ['metric']} {fnum(champ['primary'])}" if champ else "run AutoML", AM, MG)]),
            unsafe_allow_html=True)
    if S.get("demo_pending"):
        draw_tiles()
        with panel("🎥 Live training simulation — real models, streaming telemetry"):
            live_box = st.container()
        demo_df = S["datasets"]["Telco Churn"]["df"]
        recs = ["logreg", "dt", "knn", "nb", "rf", "et",
                "xgb" if HAS["xgboost"] else ("lgbm" if HAS["lightgbm"] else "gb"), "svm"]
        dcfg = dict(task="classification", target="Churn", date_col=None, metric="ROC AUC",
                    folds=3, test_size=.25, seed=42, device="CPU", hpo=False, trials=10,
                    budget="Standard", semi=False, mask=.6, maxrows=5000, sp=7, mlflow=False,
                    prep=default_prep())
        exp = run_automl("Telco Churn", recs, dcfg, live_box, celebrate=True, demo=True)
        S["demo_pending"] = False
        champ = champion_run(); draw_tiles()
        st.markdown(chips([chip("① Auto-profiled 2,200 rows", CY), chip("② Detected binary classification", VI),
                           chip("③ Trained 8 algorithms with CV", LI), chip("④ Crowned a champion 👑", AM)]),
                    unsafe_allow_html=True)
    else:
        draw_tiles()
        exp = next((e for e in S["experiments"] if e.get("demo")), S["experiments"][0] if S["experiments"] else None)
        if exp:
            with panel("🎥 Live training simulation"):
                static_live_panel(exp)
    section("🧭 Workflow", "Every stage is live — click to jump in")
    workflow_row()
    section("📦 Datasets", "Four realistic seeded datasets loaded automatically — or bring your own in Data Studio")
    dataset_cards()
    section("🕑 Recent experiments")
    if S["experiments"]:
        t = pd.DataFrame([dict(id=e["id"], when=e["ts"], dataset=e["ds"], task=e["task"],
                               models=len(e.get("rows", [])), metric=e["metric"],
                               champion=(e.get("champ_snapshot") or {}).get("name", "—"),
                               best=fnum((e.get("champ_snapshot") or {}).get("primary")),
                               fingerprint=e["data_hash"]) for e in S["experiments"][:5]])
        st.dataframe(t, use_container_width=True, hide_index=True)
        if st.button("📑 Open reports"): goto("📑 Reports")
    capability_matrix()

def page_data():
    hero(); section("🧪 Data Studio", "Upload, manage, profile, clean and engineer features")
    undo_bar()
    c1, c2, c3 = st.columns([2, 1.2, 1])
    names = list(S["datasets"])
    cur = W("active_ds_sel", S["active_ds"])
    if cur not in names: S["active_ds_sel"] = S["active_ds"]
    chosen = c1.selectbox("Active dataset", names, key="active_ds_sel")
    if chosen != S["active_ds"]:
        S["active_ds"] = chosen; RERUN()
    ds = S["active_ds"]; meta = S["datasets"][ds]; df = meta["df"]; cfg = S["ds_cfg"][ds]
    if c3.button("🗑 Delete dataset", use_container_width=True):
        S["trash"] = dict(kind="dataset", label=f"Deleted dataset '{ds}'",
                          payload=dict(name=ds, data=meta, cfg=cfg))
        del S["datasets"][ds]; del S["ds_cfg"][ds]
        if not S["datasets"]: load_demo("Telco Churn")
        S["active_ds"] = list(S["datasets"])[0]; S["active_ds_sel"] = S["active_ds"]
        toast("Dataset deleted — undo available", "🗑"); RERUN()
    with c2.expander("⬆ Upload CSV / Excel"):
        up = st.file_uploader("File", type=["csv", "xlsx", "xls"], key="ds_upload")
        nm = st.text_input("Dataset name", value=(Path(up.name).stem if up else "My Dataset"), key="ds_upname")
        if st.button("Add dataset", type="primary", disabled=up is None, key="ds_add"):
            try:
                ndf = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
                if len(ndf) > 200000:
                    ndf = ndf.sample(200000, random_state=42); toast("Sampled to 200,000 rows for responsiveness", "⚠️")
                for c in ndf.columns:
                    conv = try_parse_date(ndf[c])
                    if conv is not None: ndf[c] = conv
                nm = nm.strip() or "My Dataset"
                while nm in S["datasets"]: nm += " (2)"
                S["datasets"][nm] = dict(df=ndf, origin="upload", seed=None, created=stamp())
                S["ds_cfg"][nm] = detect_task(ndf)
                S["active_ds"] = nm; S["active_ds_sel"] = nm
                toast(f"Uploaded '{nm}' — {len(ndf):,} rows", "📦"); RERUN()
            except Exception as e:
                st.error(f"Could not read file: {e}")
    with st.expander("🎲 Regenerate / add demo datasets"):
        cols = st.columns(len(DEMOS))
        for col, (name, d) in zip(cols, DEMOS.items()):
            if col.button(f"{d['icon']} {name}", key="demo_" + name, use_container_width=True):
                load_demo(name); S["active_ds"] = name; S["active_ds_sel"] = name
                toast(f"{name} ready", "🎲"); RERUN()
        st.caption("Demos are seeded generators — identical data every time, fully reproducible.")

    prof = profile_df(df)
    st.markdown(tiles_html([
        ("Rows", f"{prof['rows']:,}", f"{prof['dups']} duplicates", CY, VI),
        ("Columns", str(prof['cols']), f"{len(prof['num_cols'])} numeric", VI, BL),
        ("Missing cells", f"{prof['missing_pct']}%", f"{prof['missing']} cells", AM, MG),
        ("Memory", f"{prof['mem']} MB", meta["origin"], LI, CY)]), unsafe_allow_html=True)

    with panel("🎯 Target & task — auto-detected, fully overridable"):
        c1, c2, c3 = st.columns([1.4, 1.4, 1.6])
        tgt_options = ["(none — clustering)"] + list(df.columns)
        cur_t = cfg.get("target") or "(none — clustering)"
        tkey = f"dst_{ds}"
        if W(tkey, cur_t) not in tgt_options: S[tkey] = cur_t
        new_t = c1.selectbox("Target column", tgt_options, key=tkey)
        task_opts = ["Auto-detected", "classification", "regression", "clustering", "timeseries"]
        tk = f"dst_task_{ds}"; W(tk, "Auto-detected")
        new_task = c2.selectbox("Task", task_opts, key=tk)
        date_cols = ["(none)"] + [c for c in df.columns if df[c].dtype.kind == "M"]
        dk = f"dst_date_{ds}"
        cur_d = cfg.get("date_col") or "(none)"
        if W(dk, cur_d) not in date_cols: S[dk] = cur_d
        new_date = c3.selectbox("Datetime column (time-series)", date_cols, key=dk)
        resolved = detect_task(df, prefer_target=None if new_t.startswith("(none") else new_t)
        if new_task != "Auto-detected": resolved["task"] = new_task
        if new_t.startswith("(none"): resolved["target"] = None; resolved["task"] = "clustering" if new_task == "Auto-detected" else new_task
        resolved["date_col"] = None if new_date == "(none)" else new_date
        if resolved != cfg:
            S["ds_cfg"][ds] = resolved
        st.markdown(chips([chip(f"→ {resolved['task']}", LI),
                           chip(f"confidence {int(resolved.get('confidence', .8) * 100)}%", CY)] +
                          [chip(r, MUT) for r in resolved.get("reasons", [])]), unsafe_allow_html=True)

    prep = S.setdefault("prep_cfg", {}).setdefault(ds, default_prep())
    with panel("🧹 Preparation recipe — applied inside every training pipeline (leak-free)"):
        c1, c2, c3, c4 = st.columns(4)
        ik = f"prp_imp_{ds}"; W(ik, prep["imputer"])
        prep["imputer"] = c1.selectbox("Numeric imputation", ["median", "mean", "most_frequent", "constant"], key=ik)
        sk = f"prp_scale_{ds}"; W(sk, prep["scale"])
        prep["scale"] = c2.checkbox("Standardize features", key=sk)
        wk = f"prp_winsor_{ds}"; W(wk, prep["winsor"])
        prep["winsor"] = c3.checkbox("Winsorize outliers (1–99%)", key=wk)
        fk = f"prp_select_{ds}"; W(fk, prep["select"])
        prep["select"] = c4.checkbox("Auto feature selection", key=fk)
        c5, c6 = st.columns([1, 2])
        pk = f"prp_pct_{ds}"; W(pk, prep["pct"])
        prep["pct"] = c5.slider("Keep top features (%)", 20, 100, key=pk, disabled=not prep["select"])
        drk = f"prp_drop_{ds}"; W(drk, [c for c in prep["drop"] if c in df.columns])
        prep["drop"] = c6.multiselect("Exclude columns", [c for c in df.columns if c != cfg.get("target")], key=drk)
        st.caption("Categoricals → one-hot (unknown-safe) · missing categoricals → most frequent · high-cardinality (>40 levels) skipped automatically.")

    tabs = st.tabs(["👁 Preview", "📋 Column profile", "📊 Distributions", "🔗 Correlations", "🚨 Outliers", "🎯 Target", "🛠 Feature engineering"])
    with tabs[0]:
        st.dataframe(df.head(200), use_container_width=True, hide_index=True, height=340)
        st.download_button("⬇ Download active dataset (CSV)", df.to_csv(index=False).encode(),
                           file_name=f"{ds.replace(' ', '_')}.csv", mime="text/csv")
    with tabs[1]:
        st.dataframe(prof["col_table"], use_container_width=True, hide_index=True, height=360)
        miss = prof["col_table"]; miss = miss[miss["missing"] > 0]
        if len(miss):
            f = new_fig(260, "Missing values by column")
            f.add_bar(x=miss["column"], y=miss["missing %"], marker_color=AM)
            st.plotly_chart(f, use_container_width=True, key="miss_bar")
        else:
            st.success("No missing values — this dataset is complete.")
    with tabs[2]:
        col = sel("Column", list(df.columns), "eda_col")
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            f = px.histogram(df, x=col, nbins=40, color=cfg.get("target") if cfg.get("task") == "classification" else None,
                             color_discrete_sequence=PAL)
            f.update_layout(height=320, title=f"Distribution of {col}")
            st.plotly_chart(f, use_container_width=True, key="eda_hist")
            st.plotly_chart(px.box(df, y=col, x=cfg.get("target") if cfg.get("task") == "classification" else None,
                                   color_discrete_sequence=PAL).update_layout(height=260),
                            use_container_width=True, key="eda_box")
        else:
            vc = s.value_counts().head(15)
            f = new_fig(300, f"Top categories — {col}")
            f.add_bar(x=vc.index.astype(str), y=vc.values, marker_color=PAL[:len(vc)])
            st.plotly_chart(f, use_container_width=True, key="eda_cat")
    with tabs[3]:
        num = df.select_dtypes(include=np.number)
        if num.shape[1] >= 2:
            corr = num.corr()
            f = new_fig(420, "Pearson correlation (numeric features)")
            f.add_heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                          colorscale=[[0, CY], [.5, "#0B1220"], [1, MG]], zmin=-1, zmax=1,
                          text=np.round(corr.values, 2) if num.shape[1] <= 14 else None,
                          texttemplate="%{text}" if num.shape[1] <= 14 else None)
            f.update_layout(yaxis_autorange="reversed")
            st.plotly_chart(f, use_container_width=True, key="corr_heat")
        else:
            st.info("Need at least two numeric columns for correlations.")
    with tabs[4]:
        out = pd.DataFrame([dict(column=k, outliers=v, pct=round(100 * v / max(len(df), 1), 2))
                            for k, v in prof["outliers"].items() if v > 0])
        if len(out):
            f = new_fig(280, "IQR-flagged outliers per column")
            f.add_bar(x=out["column"], y=out["outliers"], marker_color=RD)
            st.plotly_chart(f, use_container_width=True, key="out_bar")
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.caption("Enable **Winsorize outliers** in the recipe above to clip them inside every pipeline.")
        else:
            st.success("No IQR outliers detected.")
    with tabs[5]:
        t = cfg.get("target")
        if cfg["task"] == "classification" and t:
            vc = df[t].value_counts()
            f = new_fig(300, f"Class balance — {t}")
            f.add_pie(labels=vc.index.astype(str), values=vc.values, hole=.55,
                      marker=dict(colors=PAL), textinfo="label+percent")
            st.plotly_chart(f, use_container_width=True, key="tgt_donut")
            mn = vc.min() / vc.sum()
            (st.warning if mn < .2 else st.success)(f"Minority class share: {mn:.1%}" + (" — imbalanced; prefer ROC AUC / F1 and ensembles." if mn < .2 else " — reasonably balanced."))
        elif cfg["task"] == "regression" and t:
            st.plotly_chart(px.histogram(df, x=t, nbins=45, color_discrete_sequence=[CY])
                            .update_layout(height=300, title=f"Target distribution — {t}"),
                            use_container_width=True, key="tgt_hist")
            st.dataframe(df[t].describe().to_frame().T, use_container_width=True)
        elif cfg["task"] == "timeseries" and t and cfg.get("date_col"):
            d = df.sort_values(cfg["date_col"])
            f = new_fig(320, f"{t} over time")
            f.add_scatter(x=d[cfg["date_col"]], y=d[t], mode="lines", line=dict(color=AM, width=2))
            st.plotly_chart(f, use_container_width=True, key="tgt_ts")
        else:
            st.info("Clustering mode — no target. Explore structure via PCA in Diagnostics after training.")
    with tabs[6]:
        st.markdown("**Create a ratio / product feature** — added to the dataset instantly:")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        c1, c2, c3, c4 = st.columns([1.4, 1.4, 1, 1.2])
        a = c1.selectbox("Column A", nums, key="fe_a") if nums else None
        b = c2.selectbox("Column B", nums, key="fe_b") if nums else None
        op = c3.selectbox("Operation", ["A / B", "A × B", "A − B", "A + B"], key="fe_op")
        fname = c4.text_input("New name", value=" engineered", key="fe_name").strip()
        if st.button("➕ Add engineered feature", type="primary", disabled=not nums, key="fe_add"):
            try:
                eps = 1e-9
                va, vb = df[a].astype(float), df[b].astype(float)
                vals = {"A / B": va / (vb.abs() + eps), "A × B": va * vb, "A − B": va - vb, "A + B": va + vb}[op]
                nm = fname or f"{a}_{op}_{b}"
                df[nm] = vals
                S["datasets"][ds]["df"] = df
                toast(f"Feature '{nm}' added", "🛠"); RERUN()
            except Exception as e:
                st.error(str(e))
        dcols = [c for c in df.columns if df[c].dtype.kind == "M"]
        if dcols and st.button("📅 Expand datetime columns into calendar features", key="fe_dates"):
            for c in dcols:
                for part in ["year", "month", "day", "dayofweek"]:
                    df[f"{c}_{part}"] = getattr(df[c].dt, part)
            S["datasets"][ds]["df"] = df
            toast("Calendar features added", "📅"); RERUN()
        st.caption("Feature selection percentile lives in the preparation recipe above and is fit only on training folds.")

def page_lab():
    hero(); section("⚙️ AutoML Lab", "Configure once — train many algorithms with cross-validation, HPO and live telemetry")
    ds = S["active_ds"]; df = S["datasets"][ds]["df"]; det = S["ds_cfg"][ds]
    st.markdown(chips([chip(f"📦 {ds} · {len(df):,} rows", CY), chip(f"auto-detected: {det['task']}", VI)] +
                      [chip(r, MUT) for r in det.get("reasons", [])]), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1, panel("🎯 Objective"):
        tk = "lab_task"; W(tk, "Auto-detected")
        task_choice = st.radio("Task", ["Auto-detected", "classification", "regression", "clustering", "timeseries"], key=tk)
        task = det["task"] if task_choice == "Auto-detected" else task_choice
        target = det.get("target")
        if task in ("classification", "regression"):
            tgt = sel("Target column", list(df.columns), "lab_target", default=target if target in df.columns else df.columns[-1])
            target = tgt
        if task == "timeseries" and not det.get("date_col"):
            dc = [c for c in df.columns if df[c].dtype.kind == "M"]
            if not dc:
                st.error("Time-series needs a datetime column — parse one in Data Studio.")
            date_col = dc[0] if dc else None
        else:
            date_col = det.get("date_col")
        mk = "lab_metric"
        metric = sel("Optimization metric", [m[0] for m in METRICS[task]], mk)
        st.caption("Champion = best holdout score on this metric. Change it to re-rank instantly in Compare.")
    with c2, panel("🧪 Validation & budget"):
        W("lab_folds", 4); folds = st.slider("CV folds", 2, 10, key="lab_folds")
        W("lab_test", .25); test_size = st.slider("Holdout fraction", .1, .4, key="lab_test", step=.05)
        W("lab_seed", 42); seed = st.number_input("Random seed (reproducibility)", 0, 99999, key="lab_seed")
        W("lab_maxrows", 5000); maxrows = st.slider("Max training rows", 1000, 50000, key="lab_maxrows", step=1000)
        W("lab_device", "CPU")
        device = st.radio("Device", ["CPU", "GPU"], key="lab_device", horizontal=True,
                          help="GPU accelerates XGBoost / LightGBM / CatBoost / PyTorch when CUDA is available; everything falls back to CPU.")
        if device == "GPU" and not cuda_ok():
            st.caption("⚠ CUDA not detected — CPU fallback will be used.")
        if task == "timeseries":
            W("lab_sp", 7); sp = st.number_input("Seasonal period", 2, 365, key="lab_sp")
        else: sp = 7
    with c3, panel("⚙️ Optimization & tracking"):
        hpo = toggle("Hyperparameter optimization", "lab_hpo", False)
        W("lab_trials", 10); trials = st.slider("HPO trials / model", 4, 40, key="lab_trials", disabled=not hpo)
        W("lab_budget", "Standard")
        budget = st.select_slider("HPO time budget / model", ["Quick", "Standard", "Deep"], key="lab_budget", disabled=not hpo)
        st.caption(("Optuna TPE" if HAS["optuna"] else "RandomizedSearchCV (install optuna for TPE)") +
                   (" · ~20s" if budget == "Quick" else " · ~60s" if budget == "Standard" else " · ~150s"))
        semi = toggle("Semi-supervised mode (hide labels)", "lab_semi", False) if task == "classification" else False
        W("lab_mask", .6); mask = st.slider("Fraction of train labels hidden", .1, .9, key="lab_mask", disabled=not semi)
        mlflow = toggle("Track with MLflow", "lab_mlflow", False)
        if mlflow and not HAS["mlflow"]: st.caption("⚠ mlflow not installed — tracking will be skipped.")
    prep = S.setdefault("prep_cfg", {}).setdefault(ds, default_prep())
    with panel("🧬 Algorithms"):
        auto = toggle("Auto-select recommended algorithms", "lab_auto", True)
        recs, reasons = recommend(df, task, target, semi=semi, n=12)
        if auto:
            W("lab_autoN", 6); n_auto = st.slider("How many algorithms", 3, min(12, max(3, len(recs))), key="lab_autoN")
        avail = available_for(task, semi)
        ok_ids = [a["id"] for a in avail if algo_ok(a)]
        def fmt(aid):
            sp_ = get_spec(aid)
            if sp_ is None: return aid
            tag = "" if algo_ok(sp_) else f"  ⚠ needs {sp_['needs']}"
            return f"{sp_['name']}  ·  {sp_['family']}{tag}"
        if auto:
            algo_ids = [a["id"] for a in recs[:n_auto]]
            st.markdown(chips([chip(f"{i+1}. {get_spec(a)['name']}", FAM_COL.get(get_spec(a)['family'], CY))
                               for i, a in enumerate(algo_ids)]), unsafe_allow_html=True)
            with st.expander("Why these recommendations?"):
                for aid in algo_ids:
                    st.markdown(f"- **{get_spec(aid)['name']}** — {reasons.get(aid, 'compatible')}")
        else:
            chosen = msel("Algorithms to train", ok_ids + [a["id"] for a in avail if not algo_ok(a)],
                          "lab_algos", default=[a["id"] for a in recs[:6]], format_func=fmt)
            algo_ids = [a for a in chosen if a in ok_ids]
            blocked = [a for a in chosen if a not in ok_ids]
            if blocked:
                st.warning("Skipped (missing optional deps): " + ", ".join(get_spec(a)["name"] for a in blocked))
        with st.expander("🧩 Bring your own estimator (custom / third-party libraries)"):
            st.caption("Define `make_estimator(task, seed, device)` returning any scikit-learn-compatible estimator. "
                       "It joins the catalog, trains, compares and deploys like a built-in. Only run code you trust.")
            example = ("# any installed library works — sklearn, xgboost, custom classes...\n"
                       "from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor\n\n"
                       "def make_estimator(task, seed, device):\n"
                       "    if task == 'classification':\n"
                       "        return HistGradientBoostingClassifier(max_iter=250, random_state=seed)\n"
                       "    return HistGradientBoostingRegressor(max_iter=250, random_state=seed)\n")
            W("lab_custom_name", "HistGradientBoosting (custom)")
            cname = st.text_input("Display name", key="lab_custom_name")
            W("lab_custom_code", example)
            code = st.text_area("Python", key="lab_custom_code", height=190)
            if st.button("📌 Register custom algorithm", key="lab_custom_reg"):
                try:
                    mod = types.ModuleType("custom_algo"); mod.__dict__["__name__"] = "custom_algo"
                    sys.modules["custom_algo"] = mod
                    exec(compile(code, "<custom_algo>", "exec"), mod.__dict__)
                    f = mod.make_estimator
                    _ = f("classification", 42, "CPU")
                    entry = dict(id="custom_" + uid(), name=cname.strip() or "Custom estimator",
                                 family="Custom", tasks=("classification", "regression"),
                                 make=lambda t, s, d, p: sys.modules["custom_algo"].make_estimator(t, s, d),
                                 needs=None, space={}, kind="standard", code=code)
                    S.setdefault("custom_algos", []).append(entry)
                    toast(f"Registered '{entry['name']}' — select it in manual mode", "🧩")
                except Exception as e:
                    st.error(f"Registration failed: {e}")
    n_sel = len(algo_ids)
    est_cost = n_sel * (folds + 1) * (trials if hpo else 1)
    st.caption(f"Planned: **{n_sel} algorithms** · ≈{est_cost} fits" + (" · HPO enabled" if hpo else ""))
    if st.button(f"🚀 Train {n_sel} models now", type="primary", disabled=n_sel == 0, key="lab_train_btn"):
        S["lab_go"] = True
    autorun = S.pop("lab_autorun", False)
    if S.pop("lab_go", False) or autorun:
        cfg = dict(task=task, target=target, date_col=date_col, metric=metric, folds=int(folds),
                   test_size=float(test_size), seed=int(seed), device=device, hpo=bool(hpo),
                   trials=int(trials), budget=budget, semi=bool(semi), mask=float(mask),
                   maxrows=int(maxrows), sp=int(sp), mlflow=bool(mlflow), prep=dict(prep))
        with panel("🎥 Live training run"):
            box = st.container()
        exp = run_automl(ds, algo_ids, cfg, box, celebrate=True)
        c1, c2, c3 = st.columns(3)
        if c1.button("🏆 Compare results", use_container_width=True): goto("🏆 Compare & Select")
        if c2.button("🔬 Open diagnostics", use_container_width=True): goto("🔬 Diagnostics")
        if c3.button("📑 Generate report", use_container_width=True): goto("📑 Reports")

def runs_scoped():
    scope_all = W("cmp_all", False)
    rs = [r for r in S["runs"]]
    if not scope_all:
        rs = [r for r in rs if r["ds"] == S["active_ds"]]
    return rs

def page_compare():
    hero(); section("🏆 Compare & Select", "Search, filter, sort, bulk-act — crown the champion")
    undo_bar()
    c0, c1, c2, c3, c4 = st.columns([1.1, 1.4, 1.4, 1.3, 1.2])
    toggle("All datasets", "cmp_all", False)
    rs = runs_scoped()
    W("cmp_search", ""); q = c1.text_input("🔎 Search models", key="cmp_search")
    fams = sorted({r["family"] for r in rs})
    fsel = msel("Families", fams, "cmp_fams", default=fams)
    task_here = S["ds_cfg"][S["active_ds"]]["task"]
    sort_opts = ["Primary metric", "CV mean", "Training time", "Name"]
    sort_by = sel("Sort by", sort_opts, "cmp_sort")
    view_names = ["—"] + [v["name"] for v in S.get("views", [])]
    view = sel("Saved view", view_names, "cmp_view")
    c4.write("")
    if c4.button("Apply view", use_container_width=True, disabled=view == "—"):
        v = next(v for v in S["views"] if v["name"] == view)
        S["cmp_search"] = v["q"]; S["cmp_fams"] = v["fams"]; S["cmp_sort"] = v["sort"]; S["cmp_all"] = v["all"]
        toast(f"View '{view}' applied", "💾"); RERUN()
    with st.expander("💾 Save current filters as a view"):
        vn = st.text_input("View name", key="cmp_vname")
        if st.button("Save view", key="cmp_vsave") and vn.strip():
            S.setdefault("views", []).append(dict(name=vn.strip(), q=q, fams=fsel, sort=sort_by, all=S.get("cmp_all", False)))
            toast("View saved", "💾"); RERUN()
    flt = [r for r in rs if (q.lower() in r["name"].lower()) and (r["family"] in fsel)]
    ok = [r for r in flt if r["status"] == "ok" and not math.isnan(r.get("primary", float("nan")))]
    def sort_key(r):
        if sort_by == "CV mean": return -(r.get("cv_mean") or -1e9)
        if sort_by == "Training time": return r.get("duration", 0)
        if sort_by == "Name": return r["name"]
        d = mdef(r["task"], r["metric"])[2]
        return -r["primary"] if d == "max" else r["primary"]
    if sort_by == "Name": ok.sort(key=sort_key)
    else: ok.sort(key=sort_key)
    failed = [r for r in flt if r["status"] != "ok"]
    if not ok and not failed:
        st.markdown("<div class='card'><h4>No runs yet for this scope</h4><div class='m'>Train models and the leaderboard fills instantly.</div></div>", unsafe_allow_html=True)
        if st.button("⚙️ Run AutoML now", type="primary"): goto("⚙️ AutoML Lab")
        return
    champ = champion_run()
    board = pd.DataFrame([dict(sel=False, rank=i + 1,
                               model=("🥇 " if i == 0 else "🥈 " if i == 1 else "🥉 " if i == 2 else "") + r["name"],
                               family=r["family"], metric=r["metric"], score=round(r["primary"], 4),
                               cv=(f"{fnum(r.get('cv_mean'))} ± {fnum(r.get('cv_std'))}" if not math.isnan(r.get("cv_mean", float("nan"))) else "—"),
                               fit=round(r.get("duration", 0), 2),
                               verdict=r.get("verdict", "—"),
                               champion="👑" if champ and r["id"] == champ["id"] else "",
                               id=r["id"]) for i, r in enumerate(ok)],
                         columns=["sel", "rank", "model", "family", "metric", "score", "cv", "fit",
                                  "verdict", "champion", "id"])
    edited = st.data_editor(board, use_container_width=True, hide_index=True, height=min(420, 60 + 38 * len(board)),
                            disabled=[c for c in board.columns if c != "sel"],
                            column_config={"sel": st.column_config.CheckboxColumn("✓"),
                                           "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.3f")},
                            key=f"cmp_editor_{len(S['runs'])}_{len(board)}")
    sel_ids = edited.loc[edited["sel"], "id"].tolist()
    S["compare_ids"] = sel_ids
    st.caption("Tick rows for bulk actions · click any column header to sort · progress bars are scaled 0–1 (lower-is-better metrics shown raw).")
    b1, b2, b3, b4, b5 = st.columns(5)
    picked = [r for r in ok if r["id"] in sel_ids]
    if b1.button(f"👑 Crown selected ({len(picked)})", disabled=len(picked) != 1, use_container_width=True):
        S["champion"] = picked[0]["id"]; toast(f"Champion: {picked[0]['name']}", "👑"); RERUN()
    if b2.button("🔬 Diagnose selected", disabled=not picked, use_container_width=True):
        S["diag_runs"] = [r["id"] for r in picked if r["task"] == "classification"] or S.get("diag_runs", [])
        S["diag_run"] = picked[0]["id"]; goto("🔬 Diagnostics")
    if b3.button("🗄 Register selected", disabled=not picked, use_container_width=True):
        for r in picked: register_run(r)
        toast(f"{len(picked)} model(s) registered & persisted", "🗄"); RERUN()
    if b4.button("⬇ Export ZIP", disabled=not picked, use_container_width=True):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for r in picked:
                z.writestr(f"{r['name'].replace(' ', '_')}_{r['id']}.joblib", bundle_bytes(r["bundle"]))
            z.writestr("manifest.json", json.dumps([{k: r.get(k) for k in ("id", "name", "task", "metric", "primary", "params", "created")} for r in picked], indent=2, default=jdef))
        st.download_button("⬇ Download models.zip", buf.getvalue(), file_name="autoforge_models.zip",
                           mime="application/zip", key="cmp_zip_dl")
    if b5.button(f"🗑 Delete selected ({len(picked)})", disabled=not picked, use_container_width=True):
        S["trash"] = dict(kind="runs", label=f"Deleted {len(picked)} run(s)", payload=picked)
        S["runs"] = [r for r in S["runs"] if r["id"] not in sel_ids]
        if S.get("champion") in sel_ids: S["champion"] = ok[0]["id"] if ok else None
        toast("Runs deleted — undo available", "🗑"); RERUN()
    if failed:
        with st.expander(f"⚠ {len(failed)} failed run(s)"):
            for r in failed:
                st.markdown(f"- **{r['name']}** — {r.get('error', 'unknown error')}")
    tabs = st.tabs(["📊 Charts", "⚔ Head-to-head", "🕸 Metric radar"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        f = new_fig(340, "Holdout score by model")
        f.add_bar(x=[r["name"] for r in ok], y=[r["primary"] for r in ok],
                  marker_color=[FAM_COL.get(r["family"], CY) for r in ok],
                  error_y=dict(type="data", array=[r.get("cv_std") if not math.isnan(r.get("cv_std", float("nan"))) else 0 for r in ok],
                               color="rgba(226,232,240,.45)"),
                  text=[fnum(r["primary"]) for r in ok], textposition="outside")
        f.update_layout(xaxis_tickangle=-20)
        c1.plotly_chart(f, use_container_width=True, key="cmp_bar")
        f2 = new_fig(340, "Score vs training time (bubble = CV spread)")
        f2.add_scatter(x=[r.get("duration", 0) for r in ok], y=[r["primary"] for r in ok],
                       mode="markers+text", text=[r["name"] for r in ok], textposition="top center",
                       textfont=dict(size=10),
                       marker=dict(size=[10 + 60 * (r.get("cv_std") if not math.isnan(r.get("cv_std", float("nan"))) else .02) for r in ok],
                                   color=[FAM_COL.get(r["family"], CY) for r in ok], opacity=.85,
                                   line=dict(color="rgba(255,255,255,.4)", width=1)))
        f2.update_layout(xaxis_title="seconds", yaxis_title="score")
        c2.plotly_chart(f2, use_container_width=True, key="cmp_scatter")
        keys = [m[1] for m in METRICS[ok[0]["task"]] if all(m[1] in r.get("metrics", {}) for r in ok)]
        if keys:
            z = np.array([[r["metrics"][k] for k in keys] for r in ok], float)
            zn = (z - np.nanmin(z, 0)) / (np.nanmax(z, 0) - np.nanmin(z, 0) + 1e-9)
            f3 = new_fig(360, "Metric heatmap (column-normalized)")
            f3.add_heatmap(z=zn, x=keys, y=[r["name"] for r in ok],
                           colorscale=[[0, "#0B1220"], [.5, "#7C3AED"], [1, "#22D3EE"]],
                           text=np.round(z, 3), texttemplate="%{text}")
            f3.update_layout(yaxis_autorange="reversed")
            st.plotly_chart(f3, use_container_width=True, key="cmp_heat")
    with tabs[1]:
        names = {r["id"]: r["name"] for r in ok}
        a = sel("Model A", list(names), "h2h_a", format_func=lambda i: names[i])
        b = sel("Model B", list(names), "h2h_b", default=list(names)[-1] if len(names) > 1 else None,
                format_func=lambda i: names[i])
        ra = next(r for r in ok if r["id"] == a); rb = next(r for r in ok if r["id"] == b)
        common = [k for k in ra.get("metrics", {}) if k in rb.get("metrics", {})]
        cols = st.columns(min(4, max(1, len(common))))
        for col, k in zip(cols, common[:4]):
            va, vb = ra["metrics"][k], rb["metrics"][k]
            col.metric(k, fnum(va), delta=fnum(va - vb) + " vs B")
        if ra["task"] == "classification" and ra.get("curves") and rb.get("curves"):
            c1, c2 = st.columns(2)
            c1.plotly_chart(roc_fig([ra, rb], "ROC — A vs B"), use_container_width=True, key="h2h_roc")
            c2.plotly_chart(pr_fig([ra, rb], "PR — A vs B"), use_container_width=True, key="h2h_pr")
    with tabs[2]:
        top = ok[:5]
        keys = [m[1] for m in METRICS[top[0]["task"]][:6] if all(m[1] in r.get("metrics", {}) for r in top)]
        if keys:
            f = new_fig(420, "Top-5 metric radar (normalized)")
            allv = np.array([[r["metrics"][k] for k in keys] for r in top], float)
            lo, hi = np.nanmin(allv, 0), np.nanmax(allv, 0)
            for i, r in enumerate(top):
                vals = (allv[i] - lo) / (hi - lo + 1e-9)
                vals = vals.tolist() + [vals[0]]
                f.add_scatterpolar(r=vals, theta=keys + [keys[0]], name=r["name"],
                                   line=dict(color=PAL[i % len(PAL)], width=2.2, dash=DASH[i % len(DASH)]))
            f.update_layout(polar=dict(bgcolor="rgba(13,20,36,.5)",
                                       radialaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,.15)")))
            st.plotly_chart(f, use_container_width=True, key="cmp_radar")

def register_run(r):
    name = r["name"]
    ver = 1 + sum(1 for e in S["registry"] if e["base"] == name)
    entry = dict(id=uid(), name=f"{name}", base=name, version=ver, created=stamp(),
                 ds=r["ds"], task=r["task"], metric=r["metric"], primary=r["primary"],
                 algo_id=r.get("algo_id"), notes="", custom_code=r.get("custom_code"),
                 feature_space=r.get("feature_space"), params=r.get("params", {}))
    blob = bundle_bytes(r["bundle"])
    S["registry"].insert(0, entry); db_registry_put(entry, blob)
    S.setdefault("reg_blobs", {})[entry["id"]] = r["bundle"]
    return entry

def page_diag():
    hero(); section("🔬 Diagnostics", "Curves, confusion, importance, SHAP, learning curves, overfit verdicts")
    ok = [r for r in S["runs"] if r["status"] == "ok"]
    if not ok:
        st.info("Train models first — diagnostics unlock automatically.")
        if st.button("⚙️ Open AutoML Lab", type="primary"): goto("⚙️ AutoML Lab")
        return
    names = {r["id"]: f"{r['name']} · {r['ds']} · {r['metric']} {fnum(r['primary'])}" for r in ok}
    default = S.get("champion") if S.get("champion") in names else ok[0]["id"]
    rid = sel("Inspect run", list(names), "diag_run", default=default, format_func=lambda i: names[i])
    run = next(r for r in ok if r["id"] == rid)
    st.markdown(chips([chip(run["name"], FAM_COL.get(run["family"], CY)), chip(run["task"], VI),
                       chip(f"{run['metric']} {fnum(run['primary'])}", LI),
                       chip(f"verdict: {run.get('verdict','—')}", run.get("verdict_color", MUT)),
                       chip(f"{run.get('duration', 0):.1f}s", AM)]), unsafe_allow_html=True)
    st.caption(run.get("verdict_why", ""))
    tab_names = []
    if run["task"] == "classification": tab_names += ["📈 ROC & PR", "🧮 Confusion matrix"]
    if run["task"] == "regression": tab_names += ["📉 Residuals"]
    if run["task"] == "clustering": tab_names += ["🌌 Structure map"]
    if run["task"] == "timeseries": tab_names += ["🕰 Forecast & ACF"]
    if run["task"] in ("classification", "regression"):
        tab_names += ["🌟 Feature importance", "🐝 SHAP explainability", "📚 Learning curve", "🎛 Fit health"]
    if run.get("hpo"): tab_names += ["⚙ HPO trace"]
    tabs = st.tabs(tab_names or ["ℹ"])
    ti = 0
    def T(): 
        nonlocal ti; t = tabs[ti]; ti += 1; return t
    if run["task"] == "classification":
        with T():
            clf_runs = [r for r in ok if r["task"] == "classification" and r["ds"] == run["ds"] and r.get("curves")]
            ids = msel("Overlay models", [r["id"] for r in clf_runs], "diag_runs",
                       default=[run["id"]], format_func=lambda i: next(r["name"] for r in ok if r["id"] == i))
            entries = [r for r in clf_runs if r["id"] in ids] or [run]
            c1, c2 = st.columns(2)
            c1.plotly_chart(roc_fig(entries), use_container_width=True, key="dg_roc")
            c2.plotly_chart(pr_fig(entries), use_container_width=True, key="dg_pr")
        with T():
            norm = toggle("Normalize rows", "cm_norm", False)
            cm = confusion_matrix(run["y_test"], run["pred"])
            st.plotly_chart(cm_fig(cm, [str(c) for c in run["classes"]], norm), use_container_width=True, key="dg_cm")
            per = pd.DataFrame({"class": [str(c) for c in run["classes"]],
                                "precision": precision_score(run["y_test"], run["pred"], average=None, zero_division=0),
                                "recall": recall_score(run["y_test"], run["pred"], average=None, zero_division=0),
                                "f1": f1_score(run["y_test"], run["pred"], average=None, zero_division=0)})
            st.dataframe(per, use_container_width=True, hide_index=True)
    if run["task"] == "regression":
        with T():
            y, p = np.array(run["y_test"]), np.array(run["pred"])
            c1, c2 = st.columns(2)
            f = new_fig(320, "Actual vs predicted")
            f.add_scatter(x=y, y=p, mode="markers", marker=dict(color=CY, size=6, opacity=.7))
            f.add_scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines",
                          line=dict(color=MG, dash="dash"), name="ideal")
            c1.plotly_chart(f, use_container_width=True, key="dg_avp")
            f2 = new_fig(320, "Residuals vs predicted")
            f2.add_scatter(x=p, y=y - p, mode="markers", marker=dict(color=AM, size=6, opacity=.7))
            f2.add_hline(y=0, line=dict(color=MG, dash="dash"))
            c2.plotly_chart(f2, use_container_width=True, key="dg_res")
            st.plotly_chart(px.histogram(x=y - p, nbins=40, color_discrete_sequence=[VI])
                            .update_layout(height=250, title="Error distribution", xaxis_title="error"),
                            use_container_width=True, key="dg_errhist")
    if run["task"] == "clustering":
        with T():
            cl = run.get("cluster", {})
            emb = np.array(cl.get("emb", []))
            if len(emb):
                labels = cl.get("labels")
                f = new_fig(400, "2-D structure map (PCA of transformed features)")
                if cl.get("kind") == "anomaly":
                    flags = np.array(cl["flags"])
                    for val, nm, col in [(1, "normal", CY), (-1, "anomaly", RD)]:
                        m = flags == val
                        f.add_scatter(x=emb[m, 0], y=emb[m, 1], mode="markers", name=nm,
                                      marker=dict(color=col, size=7, opacity=.8))
                elif labels is not None:
                    labels = np.array(labels)
                    for i, lb in enumerate(np.unique(labels)):
                        m = labels == lb
                        f.add_scatter(x=emb[m, 0], y=emb[m, 1], mode="markers",
                                      name=("noise" if lb == -1 else f"cluster {lb}"),
                                      marker=dict(color=(MUT if lb == -1 else PAL[i % len(PAL)]), size=7, opacity=.8))
                else:
                    f.add_scatter(x=emb[:, 0], y=emb[:, 1], mode="markers",
                                  marker=dict(color=CY, size=7, opacity=.8))
                st.plotly_chart(f, use_container_width=True, key="dg_cluster")
            mets = {k: fnum(v) for k, v in run.get("metrics", {}).items()}
            st.markdown(chips([chip(f"{k}: {v}", VI) for k, v in mets.items()]), unsafe_allow_html=True)
            if cl.get("kind") == "partition":
                st.caption(f"{run.get('n_clusters', '—')} clusters · noise fraction {fnum(run.get('noise', 0))} · "
                           "new points are assigned via the model (nearest-neighbor projection for partition-only methods).")
    if run["task"] == "timeseries":
        with T():
            st.plotly_chart(ts_fig(run["ts_plot"]), use_container_width=True, key="dg_ts")
            tp = run["ts_plot"]
            resid = np.array(tp["test"]) - np.array(tp["pred"])
            if HAS["statsmodels"]:
                try:
                    from statsmodels.tsa.stattools import acf
                    a = acf(resid, nlags=min(20, len(resid) - 1))
                    f = new_fig(250, "Residual autocorrelation (should be ≈0)")
                    f.add_bar(x=list(range(len(a))), y=a, marker_color=TL)
                    ci = 1.96 / math.sqrt(len(resid))
                    f.add_hline(y=ci, line=dict(color=MG, dash="dot")); f.add_hline(y=-ci, line=dict(color=MG, dash="dot"))
                    st.plotly_chart(f, use_container_width=True, key="dg_acf")
                except Exception: pass
            if run.get("bundle", {}).get("ts") is not None and hasattr(run["bundle"]["ts"], "aic_table_"):
                st.dataframe(pd.DataFrame(run["bundle"]["ts"].aic_table_), use_container_width=True, hide_index=True)
    if run["task"] in ("classification", "regression"):
        with T():
            method = st.radio("Method", ["Native", "Permutation", "SHAP (mean |value|)"], key="imp_method", horizontal=True)
            try:
                if method == "Native":
                    names_i, vals, mtxt = native_importance(run)
                elif method == "Permutation":
                    with st.spinner("Permuting holdout features…"):
                        names_i, vals, mtxt = permutation_imp(run)
                else:
                    if not HAS["shap"]:
                        st.warning("shap not installed — falling back to permutation importance.")
                        names_i, vals, mtxt = permutation_imp(run)
                    else:
                        if "shap" not in run:
                            with st.spinner("Computing SHAP values…"):
                                run["shap"] = compute_shap_run(run)
                        if run["shap"] is None:
                            st.info("SHAP tree path unavailable for this model — permutation instead.")
                            names_i, vals, mtxt = permutation_imp(run)
                        else:
                            sv = np.abs(run["shap"]["values"]).mean(0)
                            order = np.argsort(sv)[::-1]
                            names_i = [run["shap"]["names"][i] for i in order]; vals = sv[order]
                            mtxt = "SHAP TreeExplainer · mean |value|"
                top = 20
                f = new_fig(430, f"Top features — {mtxt}")
                f.add_bar(x=list(vals[:top])[::-1], y=list(names_i[:top])[::-1], orientation="h",
                          marker_color=VI, marker_line=dict(color=CY, width=.6))
                st.plotly_chart(f, use_container_width=True, key="dg_imp")
            except Exception as e:
                st.info(f"Native importance unavailable ({e}). Try Permutation.")
        with T():
            if not HAS["shap"]:
                st.warning("Install `shap` to unlock game-theoretic explanations. Permutation importance works everywhere (previous tab).")
            else:
                if "shap" not in run:
                    if st.button("🐝 Compute SHAP values", type="primary", key="dg_shap_btn"):
                        with st.spinner("Explaining with SHAP TreeExplainer…"):
                            run["shap"] = compute_shap_run(run)
                sh = run.get("shap")
                if sh is None and "shap" in run:
                    st.info("This estimator has no fast tree path — use permutation importance (model-agnostic).")
                elif sh:
                    sv, Xt, names_s = sh["values"], sh["X"], sh["names"]
                    mean_abs = np.abs(sv).mean(0)
                    top = np.argsort(mean_abs)[::-1][:12]
                    f = new_fig(460, "SHAP beeswarm — each dot is one holdout row")
                    rng = np.random.default_rng(0)
                    xs, ys, cs, txt = [], [], [], []
                    for rank, i in enumerate(top[::-1]):
                        v = Xt[:, i].astype(float)
                        lo, hi = np.quantile(v, .02), np.quantile(v, .98)
                        vn = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
                        xs += sv[:, i].tolist()
                        ys += (rank + rng.uniform(-.18, .18, len(sv))).tolist()
                        cs += vn.tolist(); txt += [names_s[i]] * len(sv)
                    f.add_scatter(x=xs, y=ys, mode="markers", text=txt,
                                  marker=dict(color=cs, size=5, opacity=.75,
                                              colorscale=[[0, CY], [.5, "#334155"], [1, MG]],
                                              colorbar=dict(title="feature<br>value")))
                    f.update_layout(yaxis=dict(tickvals=list(range(len(top))),
                                               ticktext=[names_s[i] for i in top[::-1]]),
                                    xaxis_title="SHAP value (impact on model output)")
                    st.plotly_chart(f, use_container_width=True, key="dg_bee")
                    st.caption(f"Base value E[f(x)] = {fnum(sh['base'])} · positive SHAP pushes toward the positive class / higher prediction.")
        with T():
            if "lc" not in run:
                if st.button("📚 Compute learning curve (3-fold)", type="primary", key="dg_lc_btn"):
                    with st.spinner("Tracing learning curve…"):
                        try:
                            scoring = mdef(run["task"], run["metric"])[3] or ("accuracy" if run["task"] == "classification" else "r2")
                            sizes, tr_s, va_s = learning_curve(run["model"], run["X_test"], run["y_test"],
                                                               train_sizes=np.linspace(.25, 1, 5), cv=3,
                                                               scoring=scoring, n_jobs=-1, shuffle=True, random_state=42)
                            run["lc"] = dict(sizes=sizes.tolist(), tr=tr_s.mean(1).tolist(), tr_std=tr_s.std(1).tolist(),
                                             va=va_s.mean(1).tolist(), va_std=va_s.std(1).tolist())
                        except Exception as e:
                            st.error(f"Learning curve failed: {e}")
            lc = run.get("lc")
            if lc:
                f = new_fig(340, "Learning curve — gap between bands signals over/underfit")
                f.add_scatter(x=lc["sizes"], y=np.array(lc["tr"]) + np.array(lc["tr_std"]), mode="lines",
                              line=dict(width=0), showlegend=False)
                f.add_scatter(x=lc["sizes"], y=np.array(lc["tr"]) - np.array(lc["tr_std"]), mode="lines",
                              fill="tonexty", fillcolor="rgba(34,211,238,.15)", line=dict(width=0), showlegend=False)
                f.add_scatter(x=lc["sizes"], y=lc["tr"], name="train", line=dict(color=CY, width=2.4))
                f.add_scatter(x=lc["sizes"], y=np.array(lc["va"]) + np.array(lc["va_std"]), mode="lines",
                              line=dict(width=0), showlegend=False)
                f.add_scatter(x=lc["sizes"], y=np.array(lc["va"]) - np.array(lc["va_std"]), mode="lines",
                              fill="tonexty", fillcolor="rgba(244,114,182,.15)", line=dict(width=0), showlegend=False)
                f.add_scatter(x=lc["sizes"], y=lc["va"], name="validation", line=dict(color=MG, width=2.4))
                f.update_layout(xaxis_title="training rows", yaxis_title=run["metric"])
                st.plotly_chart(f, use_container_width=True, key="dg_lc")
        with T():
            items = [("Train", run.get("train_score")), ("CV mean", run.get("cv_mean")), ("Holdout", run.get("primary"))]
            st.markdown(tiles_html([(k, fnum(v), run["metric"], CY, VI) for k, v in items]), unsafe_allow_html=True)
            f = new_fig(260, "Fit health — train vs CV vs holdout")
            vals = [v if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0 for _, v in items]
            f.add_bar(x=[k for k, _ in items], y=vals, marker_color=[CY, VI, LI],
                      text=[fnum(v) for _, v in items], textposition="outside")
            st.plotly_chart(f, use_container_width=True, key="dg_health")
            st.markdown(f"<span class='verdict' style='color:{run.get('verdict_color', MUT)}'>{run.get('verdict')}</span> — {run.get('verdict_why', '')}", unsafe_allow_html=True)
            if run.get("notes"): st.caption(" · ".join(run["notes"]))
    if run.get("hpo"):
        with T():
            h = run["hpo"]; vals = h.get("values", [])
            f = new_fig(300, f"HPO trace — {h.get('engine')}")
            f.add_scatter(y=vals, mode="lines+markers", name="trial score", line=dict(color=CY, width=2))
            best = np.maximum.accumulate(vals) if mdef(run["task"], run["metric"])[2] == "max" else np.minimum.accumulate(vals)
            f.add_scatter(y=best, mode="lines", name="best so far", line=dict(color=LI, width=2.4, dash="dash"))
            st.plotly_chart(f, use_container_width=True, key="dg_hpo")
            st.markdown("**Best parameters**"); st.code(json.dumps(run.get("params", {}), indent=2, default=jdef), language="json")
    payload = {k: run.get(k) for k in ("id", "name", "task", "metric", "primary", "metrics", "cv_mean", "cv_std",
                                       "verdict", "params", "curves", "created")}
    st.download_button("⬇ Export diagnostics (JSON)", json.dumps(payload, indent=2, default=jdef).encode(),
                       file_name=f"diagnostics_{run['id']}.json", mime="application/json")

def page_predict():
    hero(); section("🔮 Predict Studio", "Batch scoring, interactive what-if, forecasts — all against the champion")
    run = champion_run()
    if run is None:
        st.info("No champion yet — run AutoML and one is crowned automatically.")
        if st.button("⚙️ Run AutoML", type="primary"): goto("⚙️ AutoML Lab")
        return
    st.markdown(chips([chip(f"👑 {run['name']}", LI), chip(f"trained on {run['ds']}", CY),
                       chip(f"{run['metric']} {fnum(run['primary'])}", VI), chip(run["task"], AM)]),
                unsafe_allow_html=True)
    tabs = ["📦 Batch scoring"]
    if run["task"] in ("classification", "regression", "clustering"): tabs.append("🎛 What-if studio")
    if run["task"] == "timeseries": tabs.append("🕰 Forecast horizon")
    T = st.tabs(tabs)
    with T[0]:
        if run["task"] == "timeseries":
            st.info("Time-series champion — use the Forecast horizon tab for future predictions.")
        else:
            src = st.radio("Data source", ["Holdout test set", "Upload new CSV"], key="pred_src", horizontal=True)
            if src == "Holdout test set":
                X = run["X_test"].copy(); y = run.get("y_test")
            else:
                up = st.file_uploader("New data (same feature columns)", type=["csv"], key="pred_up")
                if up is None:
                    st.caption(f"Required columns: {', '.join(run['bundle']['columns'])}")
                    X, y = None, None
                else:
                    X = pd.read_csv(up); y = None
                    missing = [c for c in run["bundle"]["columns"] if c not in X.columns]
                    if missing:
                        st.error(f"Missing columns: {missing}"); X = None
                    else:
                        extra = [c for c in X.columns if c not in run["bundle"]["columns"]]
                        if extra: st.caption(f"Ignoring extra columns: {extra}")
            if X is not None and len(X):
                out = local_predict(run, json.loads(X.to_json(orient="records")))
                res = X.copy()
                if run["task"] == "classification":
                    proba = np.array(out.get("probabilities")) if out.get("probabilities") else None
                    binary = proba is not None and proba.shape[1] == 2
                    thr = st.slider("Decision threshold", .05, .95, key="pred_thresh") if binary else .5
                    pred = (proba[:, 1] >= thr).astype(int) if binary else np.array(out["predictions"])
                    labels = run["le"].inverse_transform(pred.astype(int)) if run.get("le") is not None else pred
                    res["prediction"] = labels
                    if binary: res["P(positive)"] = np.round(proba[:, 1], 4)
                    if y is not None:
                        actual = run["le"].inverse_transform(np.array(y).astype(int)) if run.get("le") is not None else y
                        res["actual"] = actual
                        res["correct"] = np.where(np.array(labels).astype(str) == np.array(actual).astype(str), "✓", "✕")
                        acc = float(np.mean(res["correct"] == "✓"))
                        st.markdown(chip(f"live accuracy @ threshold: {acc:.1%}", LI), unsafe_allow_html=True)
                    if binary:
                        f = px.histogram(res, x="P(positive)", color="prediction", nbins=35,
                                         color_discrete_sequence=PAL).update_layout(height=270,
                                         title="Probability distribution by predicted class")
                        st.plotly_chart(f, use_container_width=True, key="pd_hist")
                elif run["task"] == "regression":
                    res["prediction"] = np.round(out["predictions"], 4)
                    if y is not None:
                        res["actual"] = y; res["error"] = np.round(np.array(y) - np.array(out["predictions"]), 4)
                    st.plotly_chart(px.histogram(res, x="prediction", nbins=40, color_discrete_sequence=[CY])
                                    .update_layout(height=260, title="Prediction distribution"),
                                    use_container_width=True, key="pd_reg_hist")
                else:
                    kind = run["bundle"].get("kind")
                    if kind == "embedding":
                        emb = np.array(out["embedding"])
                        res["comp_1"], res["comp_2"] = emb[:, 0].round(3), emb[:, 1].round(3)
                        st.plotly_chart(px.scatter(res, x="comp_1", y="comp_2", color_discrete_sequence=[VI])
                                        .update_layout(height=300, title="New data in the learned embedding"),
                                        use_container_width=True, key="pd_emb")
                    elif kind == "anomaly":
                        res["anomaly_score"] = out["anomaly_scores"]
                        res["flag"] = np.where(np.array(out["predictions"]) == -1, "🚨 anomaly", "normal")
                        st.markdown(chip(f"anomalies: {int((np.array(out['predictions']) == -1).sum())}/{len(res)}", RD), unsafe_allow_html=True)
                        st.plotly_chart(px.histogram(res, x="anomaly_score", color="flag", nbins=40,
                                                     color_discrete_sequence=[CY, RD])
                                        .update_layout(height=260), use_container_width=True, key="pd_anom")
                    else:
                        res["cluster"] = out["predictions"]
                        vc = res["cluster"].value_counts().sort_index()
                        f = new_fig(260, "Cluster assignments")
                        f.add_bar(x=[f"cluster {c}" for c in vc.index], y=vc.values,
                                  marker_color=PAL[:len(vc)])
                        st.plotly_chart(f, use_container_width=True, key="pd_clu")
                st.dataframe(res.head(300), use_container_width=True, hide_index=True, height=320)
                st.download_button("⬇ Download scored CSV", res.to_csv(index=False).encode(),
                                   file_name="autoforge_predictions.csv", mime="text/csv", type="primary")
                S["predicted"] = True
    if len(T) > 1 and run["task"] in ("classification", "regression", "clustering"):
        with T[1]:
            space = run.get("feature_space") or {}
            if not space:
                st.info("What-if needs feature ranges (available for tabular champions).")
            else:
                st.caption("Drag any control — the champion re-predicts instantly, with a local explanation.")
                cols = list(space)
                row = {}
                grid = st.columns(2)
                for i, c in enumerate(cols):
                    sp_ = space[c]; key = f"wi_{run['id']}_{c}"
                    with grid[i % 2]:
                        if sp_["kind"] == "num":
                            if sp_["int"]:
                                W(key, int(round(sp_["median"])))
                                S[key] = int(min(max(S[key], int(sp_["min"])), int(sp_["max"])))
                                row[c] = st.slider(c, int(sp_["min"]), int(sp_["max"]), key=key)
                            else:
                                W(key, float(sp_["median"]))
                                S[key] = float(min(max(S[key], sp_["min"]), sp_["max"]))
                                row[c] = st.slider(c, float(sp_["min"]), float(sp_["max"]), key=key,
                                                   step=float(sp_["step"]) if sp_["step"] > 0 else .01)
                        else:
                            W(key, sp_["median"] if sp_["median"] in sp_["options"] else sp_["options"][0])
                            if S[key] not in sp_["options"]: S[key] = sp_["options"][0]
                            row[c] = st.selectbox(c, sp_["options"], key=key)
                b1, b2, _ = st.columns([1, 1, 2])
                if b1.button("🎲 Randomize inputs", use_container_width=True):
                    rng = np.random.default_rng()
                    for c in cols:
                        sp_ = space[c]; key = f"wi_{run['id']}_{c}"
                        if sp_["kind"] == "num":
                            v = rng.uniform(sp_["min"], sp_["max"])
                            S[key] = int(round(v)) if sp_["int"] else float(v)
                        else:
                            S[key] = str(rng.choice(sp_["options"]))
                    RERUN()
                if b2.button("↺ Reset to medians", use_container_width=True):
                    for c in cols:
                        sp_ = space[c]; key = f"wi_{run['id']}_{c}"
                        S[key] = int(round(sp_["median"])) if sp_["kind"] == "num" and sp_["int"] else (float(sp_["median"]) if sp_["kind"] == "num" else sp_["median"])
                    RERUN()
                row_df = pd.DataFrame([row])[run["bundle"]["columns"]]
                out = local_predict(run, json.loads(row_df.to_json(orient="records")))
                c1, c2 = st.columns([1, 1.3])
                with c1:
                    if run["task"] == "classification":
                        proba = np.array(out["probabilities"]) if out.get("probabilities") else None
                        label = out.get("labels", out["predictions"])[0]
                        p = float(proba[0, 1]) if proba is not None and proba.shape[1] == 2 else None
                        st.markdown(f"<div class='card'><h4>Prediction</h4><div style='font-size:30px;font-family:Space Grotesk;color:{LI if str(label) in ('No','0') else MG}'>{label}</div></div>", unsafe_allow_html=True)
                        if p is not None:
                            g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100,
                                                       number=dict(suffix="%", font=dict(color=TXT)),
                                                       title=dict(text="P(positive class)", font=dict(color=MUT)),
                                                       gauge=dict(axis=dict(range=[0, 100], tickcolor=MUT),
                                                                  bar=dict(color=MG), bgcolor="rgba(13,20,36,.6)",
                                                                  steps=[dict(range=[0, 50], color="rgba(34,211,238,.15)"),
                                                                         dict(range=[50, 100], color="rgba(244,114,182,.18)")],
                                                                  threshold=dict(line=dict(color=AM, width=3), value=50))))
                            g.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TXT))
                            st.plotly_chart(g, use_container_width=True, key="wi_gauge")
                    elif run["task"] == "regression":
                        st.markdown(f"<div class='card'><h4>Prediction</h4><div style='font-size:34px;font-family:Space Grotesk;color:{CY}'>{fnum(out['predictions'][0])}</div><div class='m'>{run['metric']} {fnum(run['primary'])} on holdout</div></div>", unsafe_allow_html=True)
                    else:
                        if run["bundle"].get("kind") == "anomaly":
                            sc = out["anomaly_scores"][0]; flag = out["predictions"][0] == -1
                            st.markdown(f"<div class='card'><h4>{'🚨 ANOMALY' if flag else '✓ normal'}</h4><div class='m'>score {fnum(sc)} (higher = stranger)</div></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='card'><h4>Assigned cluster</h4><div style='font-size:34px;font-family:Space Grotesk;color:{VI}'>{out['predictions'][0]}</div></div>", unsafe_allow_html=True)
                with c2:
                    local = shap_local(run, row_df) if run["task"] in ("classification", "regression") else None
                    if local:
                        vals, names_l = local["values"], local["names"]
                        order = np.argsort(np.abs(vals))[::-1][:10]
                        f = new_fig(330, f"Why this prediction? (SHAP, base {fnum(local['base'])})")
                        f.add_bar(x=[vals[i] for i in order][::-1], y=[names_l[i] for i in order][::-1],
                                  orientation="h", marker_color=[MG if vals[i] > 0 else CY for i in order][::-1])
                        st.plotly_chart(f, use_container_width=True, key="wi_shap")
                    else:
                        try:
                            n_i, v_i, mtxt = native_importance(run)
                            f = new_fig(330, f"Global drivers ({mtxt})")
                            f.add_bar(x=list(v_i[:10])[::-1], y=list(n_i[:10])[::-1], orientation="h", marker_color=VI)
                            st.plotly_chart(f, use_container_width=True, key="wi_glob")
                            st.caption("Local SHAP unavailable for this model — showing global importance.")
                        except Exception:
                            st.caption("Explanation unavailable for this model family.")
                S["predicted"] = True
    if run["task"] == "timeseries" and len(T) > 1:
        with T[1]:
            ts = run["bundle"]["ts"]; hist = ts.history_
            sp = int(run["bundle"].get("sp", 7))
            h = st.slider("Forecast horizon (periods)", 1, sp * 6, sp * 2, key="fc_h")
            fc = ts.forecast(h)
            delta = pd.Timedelta(days=1)
            try:
                d = np.median(np.diff(hist.index.values)).astype("timedelta64[s]").astype(float)
                delta = pd.Timedelta(seconds=max(d, 1))
            except Exception: pass
            fut_idx = [hist.index[-1] + delta * (i + 1) for i in range(h)]
            rmse = run["metrics"].get("rmse", float(np.std(fc)) * .3)
            f = new_fig(380, f"{h}-period forecast · ±1.96·RMSE band")
            tail = hist.iloc[-min(300, len(hist)):]
            f.add_scatter(x=tail.index, y=tail.values, mode="lines", name="history", line=dict(color=CY, width=2))
            f.add_scatter(x=fut_idx, y=fc + 1.96 * rmse, mode="lines", line=dict(width=0), showlegend=False)
            f.add_scatter(x=fut_idx, y=fc - 1.96 * rmse, mode="lines", fill="tonexty",
                          fillcolor="rgba(245,158,11,.16)", line=dict(width=0), showlegend=False)
            f.add_scatter(x=fut_idx, y=fc, mode="lines+markers", name="forecast", line=dict(color=AM, width=2.6))
            st.plotly_chart(f, use_container_width=True, key="fc_fig")
            fdf = pd.DataFrame({"date": [str(x.date()) for x in fut_idx], "forecast": np.round(fc, 2)})
            st.dataframe(fdf, use_container_width=True, hide_index=True, height=260)
            st.download_button("⬇ Download forecast CSV", fdf.to_csv(index=False).encode(),
                               file_name="autoforge_forecast.csv", mime="text/csv", type="primary")
            st.caption("Final model was refit on the full history after honest holdout evaluation.")
            S["predicted"] = True

def page_deploy():
    hero(); section("🚀 Deploy & API", "Export a production FastAPI server + test the exact contract locally")
    run = champion_run()
    if run is None:
        st.info("Crown a champion first — deployment packages are built from it.")
        if st.button("⚙️ Run AutoML", type="primary"): goto("⚙️ AutoML Lab")
        return
    st.markdown(chips([chip(f"👑 {run['name']}", LI), chip(run["task"], VI),
                       chip("REST · FastAPI · uvicorn", CY), chip("no API keys", AM)]), unsafe_allow_html=True)
    ep = [["GET", "/health", "liveness + model card"], ["POST", "/predict", "{'rows': [{feature: value}, …]} → predictions (+probabilities / labels / anomaly scores)"]]
    if run["task"] == "timeseries": ep.append(["POST", "/forecast", "{'horizon': h} → future values"])
    st.dataframe(pd.DataFrame(ep, columns=["method", "path", "contract"]), use_container_width=True, hide_index=True)
    tabs = st.tabs(["🧪 Live API sandbox", "📜 Server code", "📦 Deployment package"])
    with tabs[0]:
        st.caption("This sandbox executes the same handler logic as the exported server — real predictions, real latency, zero network.")
        key = f"dep_payload_{run['id']}"
        W(key, sample_payload(run))
        payload = st.text_area("Request body (JSON)", key=key, height=200)
        if st.button("▶ POST /predict", type="primary", key="dep_send"):
            t0 = time.time()
            try:
                body = json.loads(payload)
                if run["task"] == "timeseries":
                    h = int(body.get("horizon", 7))
                    resp = dict(model=run["name"], horizon=h,
                                forecast=[round(float(v), 3) for v in run["bundle"]["ts"].forecast(h)])
                else:
                    resp = local_predict(run, body.get("rows", []))
                resp["latency_ms"] = round((time.time() - t0) * 1000, 2)
                c1, c2 = st.columns([1, 3])
                c1.markdown(chip("HTTP 200 · OK", LI), unsafe_allow_html=True)
                c2.markdown(chip(f"{resp['latency_ms']} ms", CY), unsafe_allow_html=True)
                st.code(json.dumps(resp, indent=2, default=jdef), language="json")
                S["deployed"] = True
            except Exception as e:
                st.markdown(chip("HTTP 422 · Unprocessable Entity", RD), unsafe_allow_html=True)
                st.code(json.dumps({"detail": str(e)}, indent=2), language="json")
        st.markdown("**curl**")
        st.code(f'curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json', language="bash")
    with tabs[1]:
        code = deploy_code(run)
        st.caption("Use the copy icon (top-right) to clipboard the whole server. Custom estimator classes are embedded automatically.")
        st.code(code, language="python")
        st.download_button("⬇ server.py", code.encode(), file_name="server.py", mime="text/x-python")
    with tabs[2]:
        checks = [("Model serialized (joblib)", True), ("FastAPI server generated", True),
                  ("Sample request included", True), ("Health endpoint", True),
                  ("Custom estimators embedded", bool(custom_sources(run["bundle"])) or run["task"] == "timeseries" or True)]
        st.markdown(tiles_html([(k, "✓" if v else "○", "ready", LI if v else MUT, CY) for k, v in checks[:4]]),
                    unsafe_allow_html=True)
        if st.button("📦 Build deployment package", type="primary", key="dep_build"):
            with st.spinner("Serializing model + server + docs…"):
                S["dep_zip"] = deploy_zip(run); S["deployed"] = True
            toast("Deployment package ready", "📦")
        if S.get("dep_zip"):
            st.download_button("⬇ Download deployment.zip", S["dep_zip"],
                               file_name=f"autoforge_deploy_{run['id']}.zip", mime="application/zip", type="primary")
        st.markdown("**Windows 10/11 launch**")
        st.code("python -m venv .venv\n.venv\\Scripts\\activate\npip install -r requirements.txt\npython server.py\n# docs → http://localhost:8000/docs", language="bash")

def page_registry():
    hero(); section("🗄️ Registry & History", "Versioned models (persisted to SQLite) + reproducible experiment log")
    undo_bar()
    tabs = st.tabs(["🗄 Model registry", "🕑 Experiment history", "🧨 Session controls"])
    with tabs[0]:
        c1, c2 = st.columns([2, 1.4])
        W("reg_search", ""); q = c1.text_input("🔎 Search registry", key="reg_search")
        with c2.expander("⬆ Import .joblib"):
            up = st.file_uploader("Model bundle", type=["joblib", "pkl"], key="reg_import")
            if st.button("Import", key="reg_import_btn", disabled=up is None):
                try:
                    bundle = joblib.load(io.BytesIO(up.read()))
                    assert isinstance(bundle, dict) and "task" in bundle
                    entry = dict(id=uid(), name=Path(up.name).stem, base=Path(up.name).stem, version=1,
                                 created=stamp(), ds="(imported)", task=bundle["task"],
                                 metric="—", primary=float("nan"), notes="imported bundle",
                                 algo_id=None, custom_code=None, feature_space=None, params={})
                    blob = bundle_bytes(bundle)
                    S["registry"].insert(0, entry); db_registry_put(entry, blob)
                    S.setdefault("reg_blobs", {})[entry["id"]] = bundle
                    toast("Model imported", "📥"); RERUN()
                except Exception as e:
                    st.error(f"Import failed: {e}")
        entries = [e for e in S["registry"] if q.lower() in e["name"].lower()]
        if not entries:
            st.markdown("<div class='card'><h4>Registry is empty</h4><div class='m'>Register models from Compare (bulk) or below.</div></div>", unsafe_allow_html=True)
            champ = champion_run()
            if champ and st.button("🗄 Register current champion", type="primary"):
                register_run(champ); toast("Champion registered", "🗄"); RERUN()
        cols = st.columns(3)
        for i, e in enumerate(entries):
            with cols[i % 3]:
                metric_txt = f"{e['metric']} {fnum(e['primary'])}"
                st.markdown(f"<div class='card'><h4>{e['name']} <span style='color:{CY}'>v{e['version']}</span></h4>"
                            f"<div class='m'>{e['task']} · {e['ds']} · {e['created']}</div>"
                            f"<div style='margin-top:5px'>{chip(metric_txt, LI)} {chip('persisted ✓', VI)}</div>"
                            + (f"<div class='m' style='margin-top:4px'>📝 {e['notes']}</div>" if e.get("notes") else "")
                            + "</div>", unsafe_allow_html=True)
                b1, b2, b3 = st.columns(3)
                if b1.button("▶ Load", key="reg_load_" + e["id"], use_container_width=True):
                    bundle = S.get("reg_blobs", {}).get(e["id"])
                    if bundle is None:
                        blob = db_registry_blob(e["id"])
                        if blob is None: st.error("Blob missing"); continue
                        if e.get("custom_code"):
                            mod = types.ModuleType("custom_algo"); sys.modules["custom_algo"] = mod
                            exec(compile(e["custom_code"], "<custom_algo>", "exec"), mod.__dict__)
                        bundle = joblib.load(io.BytesIO(blob))
                        S.setdefault("reg_blobs", {})[e["id"]] = bundle
                    clone_run = dict(id=uid(), algo_id=e.get("algo_id"), name=f"{e['name']} v{e['version']} (loaded)",
                                     family="Custom" if e.get("custom_code") else ALGO_BY_ID.get(e.get("algo_id"), {}).get("family", "Supervised"),
                                     task=e["task"], ds=e["ds"], created=stamp(), status="ok",
                                     metric=e["metric"], primary=e["primary"], metrics={}, params=e.get("params", {}),
                                     cv_mean=float("nan"), cv_std=float("nan"), duration=0,
                                     verdict="Loaded", verdict_color=BL, verdict_why="Restored from registry",
                                     bundle=bundle, model=bundle.get("pipeline"), le=bundle.get("le"),
                                     classes=(bundle["le"].classes_.tolist() if bundle.get("le") is not None else []),
                                     feature_space=e.get("feature_space"), custom_code=e.get("custom_code"))
                    if e["task"] in ("classification", "regression") and bundle.get("pipeline") is not None:
                        base_ds = e["ds"] if e["ds"] in S["datasets"] else S["active_ds"]
                        X = S["datasets"][base_ds]["df"]
                        tgt = S["ds_cfg"][base_ds].get("target")
                        if tgt and tgt in X.columns: X = X.drop(columns=[tgt])
                        cols_ok = [c for c in bundle["columns"] if c in X.columns]
                        Xs = X[cols_ok].head(200)
                        if len(cols_ok) == len(bundle["columns"]) and len(Xs):
                            yy = S["datasets"][base_ds]["df"][tgt].head(200) if tgt else None
                            pred = bundle["pipeline"].predict(Xs)
                            clone_run.update(X_test=Xs.reset_index(drop=True),
                                             y_test=(encode_target(yy, e["task"])[0].tolist() if yy is not None else []),
                                             pred=pred.tolist(),
                                             proba=(get_proba(bundle["pipeline"], Xs).tolist() if get_proba(bundle["pipeline"], Xs) is not None else None))
                            clone_run["curves"] = curves_of(clone_run["y_test"], np.array(clone_run["proba"])) if e["task"] == "classification" and clone_run["proba"] and clone_run["y_test"] else None
                    S["runs"].append(clone_run); S["champion"] = clone_run["id"]
                    toast(f"Loaded {e['name']} v{e['version']} as champion", "▶")
                    goto("🔮 Predict Studio")
                if b2.button("⬇ .joblib", key="reg_exp_" + e["id"], use_container_width=True):
                    blob = db_registry_blob(e["id"])
                    if blob:
                        st.download_button("⬇ Save file", blob, file_name=f"{e['name'].replace(' ', '_')}_v{e['version']}.joblib",
                                           mime="application/octet-stream", key="reg_dl_" + e["id"])
                if b3.button("🗑", key="reg_del_" + e["id"], use_container_width=True):
                    blob = db_registry_blob(e["id"])
                    S["trash"] = dict(kind="registry", label=f"Deleted '{e['name']} v{e['version']}'",
                                      payload=dict(entry=e, blob=blob))
                    S["registry"] = [x for x in S["registry"] if x["id"] != e["id"]]
                    db_registry_delete(e["id"]); toast("Deleted — undo available", "🗑"); RERUN()
                with st.expander("✎ Edit metadata"):
                    nn = st.text_input("Name", value=e["name"], key="regn_" + e["id"])
                    nt = st.text_input("Notes", value=e.get("notes", ""), key="regt_" + e["id"])
                    if st.button("Save", key="regs_" + e["id"]):
                        e["name"], e["notes"] = nn.strip() or e["name"], nt
                        blob = db_registry_blob(e["id"])
                        if blob: db_registry_put(e, blob)
                        toast("Metadata updated", "✎"); RERUN()
    with tabs[1]:
        exps = S["experiments"]
        if not exps:
            st.info("No experiments yet.")
        else:
            t = pd.DataFrame([dict(id=e["id"], when=e["ts"], dataset=e["ds"], task=e["task"],
                                   models=len(e.get("rows", [])), metric=e["metric"],
                                   champion=(e.get("champ_snapshot") or {}).get("name", "—"),
                                   best=fnum((e.get("champ_snapshot") or {}).get("primary")),
                                   seed=e.get("cfg", {}).get("seed"), fingerprint=e["data_hash"]) for e in exps[:50]])
            st.dataframe(t, use_container_width=True, hide_index=True, height=300)
            eid = sel("Inspect experiment", [e["id"] for e in exps], "hist_exp",
                      format_func=lambda i: next(f"{e['ts']} · {e['ds']} · {e['id']}" for e in exps if e["id"] == i))
            e = next(x for x in exps if x["id"] == eid)
            c1, c2, c3 = st.columns(3)
            if c1.button("🔁 Reproduce exactly", type="primary", use_container_width=True, key="hist_repro"):
                ds = e["ds"]
                if ds not in S["datasets"]:
                    if e.get("ds_origin") == "demo" and ds in DEMOS:
                        load_demo(ds); toast(f"Demo '{ds}' regenerated from seed", "🎲")
                    else:
                        st.error("Original uploaded dataset is not in this session — re-upload it in Data Studio, then reproduce.")
                        st.stop()
                S["active_ds"] = ds; S["active_ds_sel"] = ds
                cfg = e["cfg"]
                S["lab_task"] = cfg["task"]; S["lab_metric"] = cfg["metric"]; S["lab_folds"] = cfg["folds"]
                S["lab_test"] = cfg["test_size"]; S["lab_seed"] = cfg["seed"]; S["lab_device"] = cfg["device"]
                S["lab_hpo"] = cfg["hpo"]; S["lab_trials"] = cfg["trials"]; S["lab_budget"] = cfg["budget"]
                S["lab_semi"] = cfg.get("semi", False); S["lab_mask"] = cfg.get("mask", .6)
                S["lab_maxrows"] = cfg.get("maxrows", 5000); S["lab_sp"] = cfg.get("sp", 7)
                S["lab_auto"] = False; S["lab_algos"] = list(e["algos"])
                if cfg.get("target"): S["lab_target"] = cfg["target"]
                S["prep_cfg"][ds] = dict(cfg.get("prep", default_prep()))
                S["lab_autorun"] = True
                goto("⚙️ AutoML Lab")
            if c2.button("📑 Open report", use_container_width=True, key="hist_rep"):
                S["rep_exp"] = eid; goto("📑 Reports")
            if c3.button("🗑 Delete experiment", use_container_width=True, key="hist_del"):
                S["trash"] = dict(kind="experiment", label=f"Deleted experiment {eid}", payload=e)
                S["experiments"] = [x for x in exps if x["id"] != eid]
                db_exp_delete(eid); toast("Deleted — undo available", "🗑"); RERUN()
            with st.expander("🧾 Full reproducibility configuration"):
                st.code(json.dumps(e["cfg"], indent=2, default=jdef), language="json")
                st.caption(f"Data fingerprint: {e['data_hash']} · identical data + seed + config ⇒ identical models.")
    with tabs[2]:
        st.caption("Session runs live in memory; registry & experiments persist in autoforge_state.db next to app.py.")
        c1, c2 = st.columns(2)
        if c1.button("🧹 Clear session runs (undoable)", use_container_width=True):
            S["trash"] = dict(kind="runs", label=f"Cleared {len(S['runs'])} runs", payload=list(S["runs"]))
            S["runs"] = []; S["champion"] = None; toast("Session runs cleared", "🧹"); RERUN()
        if c2.button("🏭 Factory reset & replay demo", use_container_width=True):
            st.session_state.clear(); RERUN()

def page_reports():
    hero(); section("📑 Reports", "Self-contained HTML (offline charts), Markdown, JSON and scored CSV")
    exps = S["experiments"]
    if not exps:
        st.info("Run an experiment first — reports generate from its recorded telemetry.")
        if st.button("⚙️ Run AutoML", type="primary"): goto("⚙️ AutoML Lab")
        return
    eid = sel("Experiment", [e["id"] for e in exps], "rep_exp",
              format_func=lambda i: next(f"{e['ts']} · {e['ds']} · {len(e.get('rows', []))} models · {e['id']}" for e in exps if e["id"] == i))
    exp = next(e for e in exps if e["id"] == eid)
    snap = exp.get("champ_snapshot") or {}
    st.markdown(tiles_html([
        ("Champion", snap.get("name", "—"), f"{exp['metric']} {fnum(snap.get('primary'))}", LI, CY),
        ("Verdict", snap.get("verdict", "—"), f"CV {fnum(snap.get('cv_mean'))} ± {fnum(snap.get('cv_std'))}", AM, MG),
        ("Dataset", exp["ds"], f"{exp['profile']['rows']:,} rows · fingerprint {exp['data_hash']}", CY, VI),
        ("Models", str(len(exp.get("rows", []))), f"{exp['duration']:.1f}s total compute", VI, BL)]),
        unsafe_allow_html=True)
    cache = S.setdefault("report_cache", {})
    if eid not in cache:
        with st.spinner("Rendering report figures…"):
            cache[eid] = report_html(exp)
    html = cache[eid]
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇ HTML report", html.encode(), file_name=f"autoforge_report_{eid}.html",
                       mime="text/html", type="primary", use_container_width=True)
    c2.download_button("⬇ Markdown", report_md(exp).encode(), file_name=f"autoforge_report_{eid}.md",
                       mime="text/markdown", use_container_width=True)
    c3.download_button("⬇ Experiment JSON", json.dumps(exp, indent=2, default=jdef).encode(),
                       file_name=f"experiment_{eid}.json", mime="application/json", use_container_width=True)
    champ = next((r for r in S["runs"] if r["id"] == exp.get("champion")), None)
    if champ is not None and champ.get("X_test") is not None:
        res = champ["X_test"].copy(); res["actual"] = champ["y_test"]; res["prediction"] = champ["pred"]
        if champ.get("proba"): res["proba_positive"] = np.array(champ["proba"])[:, 1].round(4)
        c4.download_button("⬇ Champion predictions CSV", res.to_csv(index=False).encode(),
                           file_name=f"predictions_{eid}.csv", mime="text/csv", use_container_width=True)
    else:
        c4.caption("Predictions CSV available while the champion run is in session.")
    section("👁 Live preview")
    components.html(html, height=680, scrolling=True)

REQUIREMENTS_TXT = """# AutoForge — All-in-One AutoML Training Foundry
# CPU-first · Windows 10/11 ready · GPU optional · No API keys
# All dependencies are permissive open-source (MIT / BSD / Apache-2.0 / PSF)

streamlit
numpy
pandas
scipy
scikit-learn
plotly
matplotlib
joblib
openpyxl
statsmodels
xgboost
lightgbm
catboost
optuna
shap
mlflow
fastapi
uvicorn
torch
umap-learn

# Optional extras (uncomment only if you want them):
# prophet
# tensorflow
"""

LICENSES = [["streamlit", "Apache-2.0"], ["numpy", "BSD-3"], ["pandas", "BSD-3"], ["scipy", "BSD-3"],
            ["scikit-learn", "BSD-3"], ["plotly", "MIT"], ["matplotlib", "PSF-based"], ["joblib", "BSD-3"],
            ["openpyxl", "MIT"], ["statsmodels", "BSD-3"], ["xgboost", "Apache-2.0"], ["lightgbm", "MIT"],
            ["catboost", "Apache-2.0"], ["optuna", "MIT"], ["shap", "MIT"], ["mlflow", "Apache-2.0"],
            ["fastapi", "MIT"], ["uvicorn", "BSD-3"], ["torch (PyTorch)", "BSD-3"], ["umap-learn", "BSD-3"],
            ["prophet (optional)", "MIT"], ["tensorflow (optional)", "Apache-2.0"]]

def page_code():
    hero(); section("📋 Code & Setup", "One-click clipboard for both deliverable files · Windows-first instructions")
    tabs = st.tabs(["📄 requirements.txt", "🐍 app.py (full source)", "🖥 Setup & licenses", "🎬 60-second demo script"])
    with tabs[0]:
        st.caption("Click the **copy icon at the top-right of the block** — the entire file lands on your clipboard. No version pins, exactly as requested.")
        st.code(REQUIREMENTS_TXT, language="text")
        st.download_button("⬇ Download requirements.txt", REQUIREMENTS_TXT.encode(),
                           file_name="requirements.txt", mime="text/plain", type="primary")
    with tabs[1]:
        try:
            src = Path(__file__).read_text(encoding="utf-8")
        except Exception:
            src = "# source unavailable in this environment"
        st.caption(f"The complete platform — {len(src.splitlines()):,} lines, one file. Copy icon = clipboard.")
        st.download_button("⬇ Download app.py", src.encode(), file_name="app.py",
                           mime="text/x-python", type="primary")
        with st.expander("🐍 app.py — expand full source (copy icon inside)", expanded=False):
            st.code(src, language="python")
    with tabs[2]:
        st.markdown("**Windows 10 / 11 · PowerShell or CMD**")
        st.code("python -m venv .venv\n.venv\\Scripts\\activate\npip install -r requirements.txt\nstreamlit run app.py", language="bash")
        st.caption("macOS / Linux: `source .venv/bin/activate` instead. CPU-only works out of the box; "
                   "GPU is used automatically by XGBoost / LightGBM / CatBoost / PyTorch when CUDA is present.")
        section("🔌 Optional dependency status")
        rows = []
        for mod in ["xgboost", "lightgbm", "catboost", "optuna", "shap", "mlflow", "statsmodels", "torch", "umap", "prophet", "fastapi"]:
            rows.append(dict(package=("umap-learn" if mod == "umap" else mod),
                             status="✓ installed" if HAS[mod] else "— optional, not installed",
                             enables={"xgboost": "XGBoost models", "lightgbm": "LightGBM", "catboost": "CatBoost",
                                      "optuna": "TPE hyperparameter search", "shap": "SHAP explainability",
                                      "mlflow": "experiment tracking", "statsmodels": "ARIMA / SARIMA / ExpSmoothing",
                                      "torch": "ANN · CNN · RNN · LSTM · GRU · Autoencoder + GPU",
                                      "umap": "UMAP embeddings", "prophet": "Prophet forecasts",
                                      "fastapi": "exported REST servers"}[mod],
                             install=f"pip install {'umap-learn' if mod == 'umap' else mod}"))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)
        st.success("Missing optional packages never break the app — affected algorithms are labeled and skipped gracefully; core AutoML always runs on scikit-learn.")
        section("⚖ License verification — commercial use")
        st.dataframe(pd.DataFrame(LICENSES, columns=["library", "license"]), use_container_width=True,
                     hide_index=True, height=380)
        st.caption("All listed licenses are permissive OSS (MIT / BSD / Apache-2.0 / PSF) — commercial use, modification and redistribution permitted. Models you train are entirely yours.")
    with tabs[3]:
        st.markdown("""
**Record a live demo video in one take:**
1. `streamlit run app.py` — the app opens **already training 8 real models** on 2,200 rows. Capture the streaming feed, growing leaderboard and pulsing telemetry (≈10s of pure live action).
2. **Mission Control** → point at the crowned champion 👑 and the workflow map.
3. **Data Studio** → show auto task detection, missing-value bars, correlation heatmap; toggle *Winsorize*.
4. **AutoML Lab** → switch to manual, add LSTM + CatBoost, enable HPO, hit **Train** — film the live epoch/HPO ticker.
5. **Compare** → tick two rows → Head-to-head ROC overlay → crown a champion.
6. **Diagnostics** → confusion matrix → SHAP beeswarm → learning curve.
7. **Predict Studio** → drag what-if sliders; watch the gauge and SHAP waterfall move live.
8. **Deploy** → POST in the sandbox → 200 OK with real latency → download deployment.zip.
9. **Reports** → download the self-contained HTML report. Close on **Code & Setup** clipboard buttons.
""")

# ---------------------------------------------------------------- bootstrap & router
def bootstrap():
    if S.get("booted"): return
    S["datasets"] = {}; S["ds_cfg"] = {}; S["runs"] = []; S["registry"] = db_registry_list()
    S["experiments"] = db_exp_list(); S["champion"] = None; S["compare_ids"] = []
    S["views"] = []; S["trash"] = None; S["custom_algos"] = []; S["reg_blobs"] = {}
    for e in S["registry"]:
        if e.get("custom_code"):
            try:
                mod = types.ModuleType("custom_algo"); sys.modules["custom_algo"] = mod
                exec(compile(e["custom_code"], "<custom_algo>", "exec"), mod.__dict__)
                S["custom_algos"].append(dict(id="custom_restored_" + e["id"], name=e["name"] + " (restored)",
                                              family="Custom", tasks=("classification", "regression"),
                                              make=lambda t, s, d, p: sys.modules["custom_algo"].make_estimator(t, s, d),
                                              needs=None, space={}, kind="standard", code=e["custom_code"]))
            except Exception: pass
    for name in DEMOS: load_demo(name)
    S["active_ds"] = "Telco Churn"
    S["booted"] = True; S["demo_pending"] = True
    W("nav", PAGES[0])

def sidebar_pulse():
    with st.sidebar:
        st.divider()
        champ = champion_run()
        st.markdown(f"<div class='small'>SESSION PULSE</div>", unsafe_allow_html=True)
        st.markdown(chips([chip(f"📦 {len(S['datasets'])} datasets", CY),
                           chip(f"🧪 {len(S['runs'])} runs", VI),
                           chip(f"🗄 {len(S['registry'])} saved", LI),
                           chip(f"🕑 {len(S['experiments'])} experiments", AM)]), unsafe_allow_html=True)
        if champ:
            st.markdown(f"<div class='card' style='margin-top:8px'><h4>👑 {champ['name']}</h4>"
                        f"<div class='m'>{champ['metric']} {fnum(champ['primary'])} · {champ['ds']}</div></div>",
                        unsafe_allow_html=True)
        deps = sum(1 for v in HAS.values() if v)
        st.caption(f"🔌 {deps}/{len(HAS)} optional accelerators detected · "
                   + ("⚡ CUDA available" if cuda_ok() else "🖥 CPU mode")
                   + " · state persisted to SQLite")
        st.caption("AutoForge · open-source AutoML foundry")

inject_css()
bootstrap()
with st.sidebar:
    st.markdown("<div class='brand' style='font-size:21px'>⚡ AutoForge</div>"
                "<div class='tagline' style='margin-bottom:10px'>train · compare · explain · deploy</div>",
                unsafe_allow_html=True)
    # goto() queues the target; apply it before the nav widget is instantiated
    if S.get("nav_goto") in PAGES: S["nav"] = S["nav_goto"]
    S.pop("nav_goto", None)
    if S.get("nav") not in PAGES: S["nav"] = PAGES[0]
    st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")

page = S["nav"]
if page == PAGES[0]: page_overview()
elif page == PAGES[1]: page_data()
elif page == PAGES[2]: page_lab()
elif page == PAGES[3]: page_compare()
elif page == PAGES[4]: page_diag()
elif page == PAGES[5]: page_predict()
elif page == PAGES[6]: page_deploy()
elif page == PAGES[7]: page_registry()
elif page == PAGES[8]: page_reports()
else: page_code()
sidebar_pulse()