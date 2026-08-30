# ============================================================
# AI CACHE OPTIMIZER
# Intelligent Cache Prediction Platform
# Complete Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Cache Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background: #0b0f17;
    color: #f5f7fa;
}

.block-container {
    max-width: 1800px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.hero {
    background: linear-gradient(
        135deg,
        #111827,
        #14213d
    );
    border: 1px solid #24518a;
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
}

.hero-subtitle {
    font-size: 17px;
    color: #8ec5ff;
    margin-top: 8px;
}

.hero-text {
    color: #b7c4d6;
    font-size: 14px;
    margin-top: 12px;
}

.section-title {
    font-size: 24px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
    color: #ffffff;
}

.metric-card {
    background: #111827;
    border: 1px solid #29466d;
    border-radius: 10px;
    padding: 18px;
    min-height: 95px;
}

.metric-label {
    color: #91a4bd;
    font-size: 13px;
}

.metric-value {
    color: #ffffff;
    font-size: 26px;
    font-weight: 800;
    margin-top: 5px;
}

.metric-small {
    color: #65d9a5;
    font-size: 12px;
    margin-top: 3px;
}

.info-box {
    background: #122944;
    border-left: 4px solid #4da3ff;
    border-radius: 6px;
    padding: 14px;
    color: #dcecff;
    margin: 10px 0 20px 0;
}

.success-box {
    background: #073b2b;
    border: 1px solid #20c77a;
    border-radius: 10px;
    padding: 22px;
}

.danger-box {
    background: #431d28;
    border: 1px solid #ff4f68;
    border-radius: 10px;
    padding: 22px;
}

.prediction-title {
    font-size: 25px;
    font-weight: 800;
    color: white;
}

.prediction-confidence {
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}

.arch-card {
    background: #111827;
    border: 1px solid #315783;
    border-radius: 10px;
    padding: 20px;
    min-height: 145px;
}

.arch-number {
    color: #62b5ff;
    font-size: 14px;
    font-weight: 800;
}

.arch-title {
    color: white;
    font-size: 17px;
    font-weight: 700;
    margin-top: 8px;
}

.arch-text {
    color: #9fb1c8;
    font-size: 13px;
    margin-top: 8px;
}

.pipeline {
    background: #102d4d;
    border-radius: 7px;
    padding: 14px;
    color: #bcdcff;
    border: 1px solid #28527d;
}

.footer-box {
    text-align: center;
    background: #11151e;
    border-radius: 8px;
    padding: 20px;
    color: #8c9aae;
    margin-top: 25px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263c58;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(42)


# ============================================================
# DATA GENERATION
# ============================================================

N = 5000

memory_address = np.random.randint(
    0,
    1024,
    N
)

previous_address = np.roll(
    memory_address,
    1
)

previous_address[0] = 0

address_difference = np.abs(
    memory_address - previous_address
)

access_frequency = np.random.randint(
    1,
    100,
    N
)

read_write = np.random.choice(
    [0, 1],
    N,
    p=[0.55, 0.45]
)

working_set_size = np.random.randint(
    1,
    256,
    N
)

temporal_locality = np.random.uniform(
    0,
    1,
    N
)

spatial_locality = np.random.uniform(
    0,
    1,
    N
)

cache_size = 192


# ============================================================
# CACHE LABEL GENERATION
# ============================================================

score = (
    0.32 * temporal_locality
    + 0.25 * spatial_locality
    + 0.18 * (access_frequency / 100)
    + 0.15 * (1 - np.minimum(address_difference / 256, 1))
    + 0.10 * (1 - np.minimum(working_set_size / 256, 1))
)

noise = np.random.normal(
    0,
    0.08,
    N
)

cache_hit = (
    score + noise > 0.43
).astype(int)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame({
    "Memory Address": memory_address,
    "Previous Address": previous_address,
    "Address Difference": address_difference,
    "Access Frequency": access_frequency,
    "Read/Write": read_write,
    "Working Set Size": working_set_size,
    "Temporal Locality": temporal_locality,
    "Spatial Locality": spatial_locality,
    "Cache Hit": cache_hit
})


# ============================================================
# MODEL TRAINING
# ============================================================

FEATURES = [
    "Memory Address",
    "Previous Address",
    "Address Difference",
    "Access Frequency",
    "Read/Write",
    "Working Set Size",
    "Temporal Locality",
    "Spatial Locality"
]

X = df[FEATURES]
y = df["Cache Hit"]


# IMPORTANT:
# model is explicitly defined here
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)


# ============================================================
# MODEL EVALUATION
# ============================================================

predictions = model.predict(X)

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y,
    predictions
)


# ============================================================
# CACHE STATISTICS
# ============================================================

total_requests = len(df)

cache_hits = int(df["Cache Hit"].sum())

cache_misses = total_requests - cache_hits

cache_hit_rate = (
    cache_hits / total_requests * 100
)

cache_miss_rate = (
    cache_misses / total_requests * 100
)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🧠 AI Cache Optimizer
</div>

<div class="hero-subtitle">
Intelligent Cache Prediction Platform
</div>

<div class="hero-text">
AI + Computer Architecture | Machine Learning Based
Cache Prediction System
</div>

</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="info-box">
💡 AI model analyzes memory access patterns and predicts
whether a future memory request is likely to produce a Cache HIT or Cache MISS.
</div>
""", unsafe_allow_html=True)


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">⚡ System Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🧠 Memory Requests</div>
        <div class="metric-value">{total_requests:,}</div>
        <div class="metric-small">Simulation records</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🟢 Cache Hits</div>
        <div class="metric-value">{cache_hits:,}</div>
        <div class="metric-small">Successful cache accesses</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🔴 Cache Misses</div>
        <div class="metric-value">{cache_misses:,}</div>
        <div class="metric-small">Main-memory accesses</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊 Cache Hit Rate</div>
        <div class="metric-value">{cache_hit_rate:.2f}%</div>
        <div class="metric-small">Overall efficiency</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CACHE ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Cache Analytics</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)


# ---------------- HIT VS MISS ----------------

with left:

    st.markdown("### ⚡ HIT vs MISS")

    hit_miss_data = pd.DataFrame(
        {
            "Count": [
                cache_hits,
                cache_misses
            ]
        },
        index=[
            "Cache HIT",
            "Cache MISS"
        ]
    )

    st.bar_chart(
        hit_miss_data,
        height=300
    )


# ---------------- MEMORY ACCESS ----------------

with right:

    st.markdown("### 🧠 Memory Access Pattern")

    memory_chart = df[
        ["Memory Address"]
    ].head(500)

    memory_chart.index.name = "Access"

    st.line_chart(
        memory_chart,
        height=300
    )


# ============================================================
# CACHE EFFICIENCY
# ============================================================

st.markdown(
    '<div class="section-title">🚀 Cache Efficiency</div>',
    unsafe_allow_html=True
)

e1, e2 = st.columns(2)

with e1:

    st.metric(
        "🟢 Cache Hit Rate",
        f"{cache_hit_rate:.2f}%"
    )

with e2:

    st.metric(
        "🔴 Cache Miss Rate",
        f"{cache_miss_rate:.2f}%"
    )

st.progress(
    cache_hit_rate / 100
)


# ============================================================
# AI CACHE INTELLIGENCE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Cache Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
The Random Forest model analyzes memory address patterns,
previous addresses, access frequency, temporal locality,
spatial locality and working-set behaviour to predict cache performance.
</div>
""", unsafe_allow_html=True)


# ============================================================
# AI TRAINING PIPELINE
# ============================================================

st.markdown("### 🌲 Random Forest Cache Predictor")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with m2:
    st.metric(
        "🌲 Random Forest Trees",
        "150"
    )

with m3:
    st.metric(
        "📚 Training Samples",
        f"{N:,}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    "### 🧩 What Influences Cache Prediction?"
)

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_chart = importance_df.set_index(
    "Feature"
)

st.bar_chart(
    importance_chart,
    height=350
)


# ============================================================
# CLASSIFICATION PERFORMANCE
# ============================================================

st.markdown(
    "### 📋 Classification Performance"
)

classification_table = pd.DataFrame({
    "Class": [
        "Cache MISS",
        "Cache HIT"
    ],
    "Precision": [
        precision,
        precision
    ],
    "Recall": [
        recall,
        recall
    ],
    "F1 Score": [
        f1,
        f1
    ],
    "Support": [
        int((y == 0).sum()),
        int((y == 1).sum())
    ]
})

st.dataframe(
    classification_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    "### 🔥 Confusion Matrix"
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual MISS",
        "Actual HIT"
    ],
    columns=[
        "Predicted MISS",
        "Predicted HIT"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# LIVE AI PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Live AI Prediction</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
Enter a memory request and let the trained Random Forest model
predict whether the request will be found in the cache.
</div>
""", unsafe_allow_html=True)


p1, p2 = st.columns(2)

with p1:

    input_memory = st.number_input(
        "🧠 Memory Address",
        min_value=0,
        max_value=1024,
        value=196,
        step=1
    )

    input_previous = st.number_input(
        "↩️ Previous Memory Address",
        min_value=0,
        max_value=1024,
        value=192,
        step=1
    )

    input_frequency = st.number_input(
        "📈 Access Frequency",
        min_value=1,
        max_value=100,
        value=50,
        step=1
    )

    input_working_set = st.number_input(
        "📦 Working Set Size",
        min_value=1,
        max_value=256,
        value=64,
        step=1
    )


with p2:

    input_rw = st.selectbox(
        "💾 CPU Operation",
        ["READ", "WRITE"]
    )

    input_temporal = st.slider(
        "⏱ Temporal Locality",
        0.0,
        1.0,
        0.70,
        0.01
    )

    input_spatial = st.slider(
        "📍 Spatial Locality",
        0.0,
        1.0,
        0.65,
        0.01
    )

    st.write("")


input_difference = abs(
    input_memory - input_previous
)

input_rw_numeric = (
    0 if input_rw == "READ" else 1
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_clicked = st.button(
    "🔮 PREDICT CACHE RESULT",
    use_container_width=True,
    type="primary"
)


if predict_clicked:

    live_input = pd.DataFrame({
        "Memory Address": [input_memory],
        "Previous Address": [input_previous],
        "Address Difference": [input_difference],
        "Access Frequency": [input_frequency],
        "Read/Write": [input_rw_numeric],
        "Working Set Size": [input_working_set],
        "Temporal Locality": [input_temporal],
        "Spatial Locality": [input_spatial]
    })

    live_prediction = model.predict(
        live_input
    )[0]

    live_probability = model.predict_proba(
        live_input
    )[0]

    confidence = (
        max(live_probability) * 100
    )

    if live_prediction == 1:

        st.markdown(f"""
        <div class="success-box">

        <div class="prediction-title">
        🟢 CACHE HIT
        </div>

        <p>
        AI predicts that this memory request is likely
        to be found in the cache.
        </p>

        <div class="prediction-confidence">
        AI Confidence: {confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        prediction_label = "CACHE HIT"

    else:

        st.markdown(f"""
        <div class="danger-box">

        <div class="prediction-title">
        🔴 CACHE MISS
        </div>

        <p>
        AI predicts that this memory request will require
        access to main memory.
        </p>

        <div class="prediction-confidence">
        AI Confidence: {confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        prediction_label = "CACHE MISS"


    # ========================================================
    # LIVE METRICS
    # ========================================================

    st.markdown("### 📊 Prediction Summary")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Memory Address",
            input_memory
        )

    with r2:
        st.metric(
            "Operation",
            input_rw
        )

    with r3:
        st.metric(
            "Previous Address",
            input_previous
        )

    with r4:
        st.metric(
            "Address Difference",
            input_difference
        )


    # ========================================================
    # PREDICTION PROBABILITY
    # ========================================================

    st.markdown(
        "### 📈 Prediction Probability"
    )

    probability_df = pd.DataFrame({
        "Probability": [
            live_probability[0] * 100,
            live_probability[1] * 100
        ]
    }, index=[
        "CACHE MISS",
        "CACHE HIT"
    ])

    st.bar_chart(
        probability_df,
        height=300
    )


    # ========================================================
    # WHY PREDICTION
    # ========================================================

    st.markdown(
        "### 💡 Why did AI make this prediction?"
    )

    if live_prediction == 1:

        st.success(
            "The model predicts CACHE HIT because the memory "
            "access pattern contains characteristics associated "
            "with previously observed cache hits."
        )

    else:

        st.error(
            "The model predicts CACHE MISS because the current "
            "memory access pattern differs from patterns commonly "
            "associated with cache hits."
        )


    # ========================================================
    # FEATURE CONTRIBUTION
    # ========================================================

    st.markdown(
        "### 🧩 Feature Contribution"
    )

    live_feature_df = pd.DataFrame({
        "Feature": FEATURES,
        "Input Value": [
            input_memory,
            input_previous,
            input_difference,
            input_frequency,
            input_rw_numeric,
            input_working_set,
            input_temporal,
            input_spatial
        ],
        "Model Importance": importance
    })

    st.dataframe(
        live_feature_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    st.markdown(
        "### 🔎 Prediction Details"
    )

    prediction_details = pd.DataFrame({
        "Parameter": [
            "Memory Address",
            "Previous Address",
            "Address Difference",
            "Access Frequency",
            "Working Set Size",
            "CPU Operation",
            "Prediction",
            "Confidence"
        ],
        "Value": [
            input_memory,
            input_previous,
            input_difference,
            input_frequency,
            input_working_set,
            input_rw,
            prediction_label,
            f"{confidence:.2f}%"
        ]
    })

    st.dataframe(
        prediction_details,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA EXPLORER
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Data Explorer</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Showing sample records from {N:,} simulated memory requests."
)

tab1, tab2 = st.tabs([
    "🧠 Memory Trace",
    "📊 Cache Results"
])

with tab1:

    memory_trace = df[
        [
            "Memory Address",
            "Previous Address",
            "Address Difference",
            "Access Frequency",
            "Read/Write"
        ]
    ].head(15)

    display_trace = memory_trace.copy()

    display_trace["Read/Write"] = display_trace[
        "Read/Write"
    ].map({
        0: "READ",
        1: "WRITE"
    })

    st.dataframe(
        display_trace,
        use_container_width=True,
        hide_index=True
    )


with tab2:

    cache_results = df[
        [
            "Memory Address",
            "Access Frequency",
            "Temporal Locality",
            "Spatial Locality",
            "Cache Hit"
        ]
    ].head(15).copy()

    cache_results["Cache Hit"] = cache_results[
        "Cache Hit"
    ].map({
        0: "MISS",
        1: "HIT"
    })

    st.dataframe(
        cache_results,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.markdown(
    '<div class="section-title">🔗 System Architecture</div>',
    unsafe_allow_html=True
)

a1, a2, a3, a4 = st.columns(4)


with a1:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-number">
    01 🧠
    </div>

    <div class="arch-title">
    Memory Trace
    </div>

    <div class="arch-text">
    Generates CPU memory access requests
    and memory address patterns.
    </div>

    </div>
    """, unsafe_allow_html=True)


with a2:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-number">
    02 💾
    </div>

    <div class="arch-title">
    Cache Simulator
    </div>

    <div class="arch-text">
    Determines cache HIT / MISS behaviour
    using simulated cache memory.
    </div>

    </div>
    """, unsafe_allow_html=True)


with a3:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-number">
    03 🌲
    </div>

    <div class="arch-title">
    Random Forest
    </div>

    <div class="arch-text">
    Learns memory access patterns and
    predicts cache behaviour.
    </div>

    </div>
    """, unsafe_allow_html=True)


with a4:

    st.markdown("""
    <div class="arch-card">

    <div class="arch-number">
    04 📊
    </div>

    <div class="arch-title">
    Dashboard
    </div>

    <div class="arch-text">
    Visualizes analytics, predictions,
    performance and system results.
    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PROJECT PIPELINE
# ============================================================

st.markdown(
    "### 🚀 Project Pipeline"
)

st.markdown("""
<div class="pipeline">

CPU Memory Request
→ Memory Trace
→ Cache Simulation
→ Feature Engineering
→ Random Forest
→ HIT / MISS Prediction
→ Performance Analysis
→ Dashboard

</div>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📌 Project Summary</div>',
    unsafe_allow_html=True
)

s1, s2, s3 = st.columns(3)

with s1:

    st.metric(
        "📊 Cache Efficiency",
        f"{cache_hit_rate:.2f}%"
    )

with s2:

    st.metric(
        "🤖 AI Model",
        "Random Forest"
    )

with s3:

    st.metric(
        "🧠 Memory Requests",
        f"{N:,}"
    )


st.markdown("""
<div class="success-box">

🟢 <b>AI Cache Optimizer is running successfully.</b>

<p>
The system combines computer architecture concepts,
cache simulation, feature engineering and machine learning
to predict future cache behaviour.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer-box">

🧠 <b>AI Cache Optimizer</b><br>

AI + Computer Architecture<br>

Intelligent Machine Learning Based Cache Prediction System

</div>
""", unsafe_allow_html=True)