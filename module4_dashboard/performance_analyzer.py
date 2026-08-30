# ============================================================
# MODULE 4 - AI CACHE OPTIMIZER DASHBOARD
# AI + COMPUTER ARCHITECTURE
# ============================================================

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Cache Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}


/* Main background */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99,102,241,0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(6,182,212,0.10),
            transparent 30%
        ),
        #f7f9fc;
}


/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* Main container */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Hero */

.hero {
    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            #eef2ff 55%,
            #ecfeff 100%
        );

    border: 1px solid #e2e8f0;

    border-radius: 28px;

    padding: 35px 40px;

    margin-bottom: 25px;

    box-shadow:
        0 15px 40px rgba(15,23,42,0.08);
}


.hero-title {
    font-size: 42px;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #0891b2
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 5px;
}


.hero-subtitle {
    color: #64748b;

    font-size: 17px;

    margin-bottom: 18px;
}


.badge {
    display: inline-block;

    background: #eef2ff;

    color: #4338ca;

    padding: 7px 14px;

    border-radius: 999px;

    font-size: 13px;

    font-weight: 700;

    margin-right: 8px;
}


/* Metric cards */

.metric-card {

    background: rgba(255,255,255,0.90);

    border: 1px solid #e2e8f0;

    border-radius: 20px;

    padding: 22px;

    min-height: 130px;

    box-shadow:
        0 10px 25px rgba(15,23,42,0.06);

    transition: 0.2s;
}


.metric-card:hover {

    transform: translateY(-3px);

    box-shadow:
        0 15px 30px rgba(15,23,42,0.10);
}


.metric-label {

    color: #64748b;

    font-size: 14px;

    font-weight: 600;
}


.metric-value {

    color: #0f172a;

    font-size: 30px;

    font-weight: 800;

    margin-top: 8px;
}


.metric-description {

    color: #94a3b8;

    font-size: 12px;

    margin-top: 5px;
}


/* Section */

.section-title {

    font-size: 24px;

    font-weight: 800;

    color: #0f172a;

    margin-top: 30px;

    margin-bottom: 15px;
}


/* Architecture cards */

.arch-card {

    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 22px;

    text-align: center;

    min-height: 150px;

    box-shadow:
        0 8px 20px rgba(15,23,42,0.05);
}


.arch-icon {

    font-size: 38px;

    margin-bottom: 8px;
}


.arch-title {

    font-size: 17px;

    font-weight: 800;

    color: #0f172a;
}


.arch-text {

    font-size: 12px;

    color: #64748b;

    margin-top: 6px;
}


.arrow {

    text-align: center;

    font-size: 32px;

    color: #6366f1;

    padding-top: 45px;
}


/* AI card */

.ai-card {

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #ecfeff
        );

    border: 1px solid #c7d2fe;

    border-radius: 24px;

    padding: 30px;

    box-shadow:
        0 12px 30px rgba(79,70,229,0.10);
}


/* Footer */

.footer {

    text-align: center;

    color: #94a3b8;

    font-size: 12px;

    margin-top: 45px;

    padding-top: 20px;

    border-top: 1px solid #e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FIND PROJECT DIRECTORY
# ============================================================

project_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


data_folder = os.path.join(
    project_folder,
    "data"
)


memory_file = os.path.join(
    data_folder,
    "memory_trace.csv"
)


cache_file = os.path.join(
    data_folder,
    "cache_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    memory_data = pd.read_csv(
        memory_file
    )

    cache_data = pd.read_csv(
        cache_file
    )

    return memory_data, cache_data


memory_data, cache_data = load_data()


# ============================================================
# CREATE AI FEATURES
# ============================================================

data = cache_data.copy()


data["operation_code"] = data[
    "operation"
].map({
    "READ": 0,
    "WRITE": 1
})


data["previous_address"] = data[
    "memory_address"
].shift(1)


data["previous_address"] = data[
    "previous_address"
].fillna(
    data["memory_address"]
)


data["address_difference"] = (
    data["memory_address"]
    - data["previous_address"]
).abs()


data["access_frequency"] = (
    data.groupby(
        "memory_address"
    ).cumcount()
)


data["target"] = data[
    "cache_result"
].map({
    "MISS": 0,
    "HIT": 1
})


# ============================================================
# TRAIN AI
# ============================================================

features = [
    "memory_address",
    "operation_code",
    "previous_address",
    "address_difference",
    "access_frequency"
]


X = data[features]

y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


report = classification_report(
    y_test,
    predictions,
    target_names=["MISS", "HIT"],
    output_dict=True,
    zero_division=0
)


matrix = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# CACHE STATISTICS
# ============================================================

total_accesses = len(cache_data)

cache_hits = (
    cache_data["cache_result"]
    .eq("HIT")
    .sum()
)


cache_misses = (
    cache_data["cache_result"]
    .eq("MISS")
    .sum()
)


hit_rate = (
    cache_hits / total_accesses
) * 100


miss_rate = (
    cache_misses / total_accesses
) * 100


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 AI Cache Lab"
    )

    st.caption(
        "Computer Architecture + Machine Learning"
    )

    st.divider()

    st.markdown(
        "### ⚙️ System Configuration"
    )

    st.info(
        "Cache Policy: LRU\n\n"
        "Cache Capacity: 16 entries\n\n"
        "Memory Accesses: 5,000"
    )

    st.divider()

    st.markdown(
        "### 📌 Modules"
    )

    st.markdown(
        "🟢 Module 1 — Memory Trace"
    )

    st.markdown(
        "🟢 Module 2 — LRU Cache"
    )

    st.markdown(
        "🟢 Module 3 — AI Predictor"
    )

    st.markdown(
        "🔵 Module 4 — Dashboard"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🧠 AI Cache Optimizer
</div>

<div class="hero-subtitle">
Intelligent Cache Performance Analysis using
Machine Learning + Computer Architecture
</div>

<span class="badge">CPU CACHE</span>

<span class="badge">LRU</span>

<span class="badge">RANDOM FOREST</span>

<span class="badge">5,000 ACCESSES</span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# KPI CARDS
# ============================================================

st.markdown(
    '<div class="section-title">⚡ Cache Performance</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-label">
        MEMORY ACCESSES
        </div>

        <div class="metric-value">
        {total_accesses:,}
        </div>

        <div class="metric-description">
        Total CPU requests
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-label">
        CACHE HITS
        </div>

        <div class="metric-value">
        {cache_hits:,}
        </div>

        <div class="metric-description">
        Served from cache
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-label">
        CACHE MISSES
        </div>

        <div class="metric-value">
        {cache_misses:,}
        </div>

        <div class="metric-description">
        Required memory access
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-label">
        HIT RATE
        </div>

        <div class="metric-value">
        {hit_rate:.2f}%
        </div>

        <div class="metric-description">
        Cache efficiency
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ARCHITECTURE FLOW
# ============================================================

st.markdown(
    '<div class="section-title">🖥️ Computer Architecture Flow</div>',
    unsafe_allow_html=True
)


a1, ar1, a2, ar2, a3 = st.columns(
    [2, 0.6, 2, 0.6, 2]
)


with a1:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-icon">🖥️</div>

    <div class="arch-title">
    CPU
    </div>

    <div class="arch-text">
    Generates memory requests
    </div>

    </div>
    """, unsafe_allow_html=True)


with ar1:

    st.markdown(
        '<div class="arrow">→</div>',
        unsafe_allow_html=True
    )


with a2:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-icon">⚡</div>

    <div class="arch-title">
    LRU CACHE
    </div>

    <div class="arch-text">
    16-entry fast memory
    </div>

    </div>
    """, unsafe_allow_html=True)


with ar2:

    st.markdown(
        '<div class="arrow">→</div>',
        unsafe_allow_html=True
    )


with a3:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-icon">💾</div>

    <div class="arch-title">
    MAIN MEMORY
    </div>

    <div class="arch-text">
    Handles cache misses
    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CACHE CHARTS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Cache Analytics</div>',
    unsafe_allow_html=True
)


chart1, chart2 = st.columns(2)


with chart1:

    st.markdown("### HIT vs MISS")

    chart_data = pd.DataFrame({
        "Result": [
            "Cache Hit",
            "Cache Miss"
        ],

        "Count": [
            cache_hits,
            cache_misses
        ]
    })

    st.bar_chart(
        chart_data.set_index("Result")
    )


with chart2:

    st.markdown("### 📈 Memory Address Pattern")

    address_data = memory_data[
        "memory_address"
    ].head(300)

    st.line_chart(
        address_data
    )


# ============================================================
# AI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Prediction Engine</div>',
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="ai-card">

    <h2>
    Random Forest Cache Predictor
    </h2>

    <p>
    The AI analyzes memory access patterns and
    predicts whether an access is likely to result
    in a cache HIT or MISS.
    </p>

    <h1>
    {accuracy * 100:.2f}% Accuracy
    </h1>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# AI METRICS
# ============================================================

st.markdown("### 🎯 AI Classification Performance")


ai1, ai2, ai3, ai4 = st.columns(4)


with ai1:

    st.metric(
        "MISS Precision",
        f"{report['MISS']['precision'] * 100:.1f}%"
    )


with ai2:

    st.metric(
        "MISS Recall",
        f"{report['MISS']['recall'] * 100:.1f}%"
    )


with ai3:

    st.metric(
        "HIT Precision",
        f"{report['HIT']['precision'] * 100:.1f}%"
    )


with ai4:

    st.metric(
        "HIT Recall",
        f"{report['HIT']['recall'] * 100:.1f}%"
    )


# ============================================================
# CONFUSION MATRIX + FEATURE IMPORTANCE
# ============================================================

cm_col, fi_col = st.columns(2)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

with cm_col:

    st.markdown("### 🔥 Confusion Matrix")

    fig, ax = plt.subplots()

    ax.imshow(matrix)

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "MISS",
        "HIT"
    ])

    ax.set_yticklabels([
        "MISS",
        "HIT"
    ])

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

with fi_col:

    st.markdown("### 🧩 AI Feature Importance")

    importance = pd.DataFrame({

        "Feature": features,

        "Importance":
            model.feature_importances_

    }).sort_values(
        "Importance",
        ascending=True
    )

    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Memory Trace Explorer</div>',
    unsafe_allow_html=True
)


with st.expander(
    "View recent memory accesses"
):

    st.dataframe(
        memory_data.head(100),
        use_container_width=True
    )


with st.expander(
    "View cache simulation results"
):

    st.dataframe(
        cache_data.head(100),
        use_container_width=True
    )


# ============================================================
# FINAL STATUS
# ============================================================

st.success(
    "✅ All four project modules are connected successfully."
)


st.markdown("""
<div class="footer">

AI Cache Optimizer • AI + Computer Architecture

Memory Trace → LRU Cache → AI Prediction → Analytics

</div>
""", unsafe_allow_html=True)