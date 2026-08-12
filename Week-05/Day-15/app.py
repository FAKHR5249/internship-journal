"""
Customer Segmentation Dashboard — Day 2
=========================================
Built with Streamlit. Loads the final KMeans model + segmented customer data
produced in Module 6 (clustering) and Module 8 (business insights).

>>> HOW TO RUN <<<
1. Copy this file into your project root (same level as your data/models folders),
   OR edit the paths in the CONFIG block below to point at your actual files.
2. pip install -r requirements.txt
3. streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ============================================================
# CONFIG — edit these paths to match your project structure
# ============================================================
DATA_PATH = "segmented_customers.csv"                 # final labeled customer data
CLUSTER_PROFILE_PATH = "cluster_profile.csv"           # avg stats per cluster
CLUSTER_NAMES_PATH = "business_cluster_names.csv"      # Cluster -> business name mapping
KMEANS_EVAL_PATH = "kmeans_evaluation.csv"              # silhouette/DB scores etc.
MODEL_PATH = "models/kmeans_model.pkl"
SCALER_PATH = "models/standard_scaler.pkl"
PIPELINE_PATH = "pipeline/customer_segmentation_pipeline.pkl"

# NOTE: this order matches kmeans_model.feature_names_in_ exactly — do not reorder,
# the scaler/model were fit on columns in this exact sequence.
FEATURES = [
    "Customer_Age", "Income", "Total_Spending", "Recency", "Customer_Tenure",
    "Family_Size", "Total_Children", "Purchase_Frequency", "Accepted_Campaigns",
    "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases",
]

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", page_icon="📊")

# ============================================================
# THEME — dark navy / cyan palette
# ============================================================
BG = "#0B1220"
CARD = "#111C2E"
ACCENT = "#00C2FF"
ACCENT2 = "#1E90FF"
TEXT = "#FFFFFF"
SUBTEXT = "#A7B3C5"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {CARD};
        border-right: 1px solid #1C2A40;
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT};
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        color: {TEXT} !important;
    }}
    h1 {{
        border-bottom: 2px solid {ACCENT};
        padding-bottom: 8px;
    }}

    /* Body / secondary text */
    p, span, label, .stMarkdown, .stCaption {{
        color: {SUBTEXT};
    }}

    /* KPI metric cards */
    div[data-testid="stMetric"] {{
        background-color: {CARD};
        border: 1px solid #1C2A40;
        border-left: 4px solid {ACCENT};
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {SUBTEXT} !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {ACCENT} !important;
    }}

    /* Tables / dataframes */
    div[data-testid="stDataFrame"] {{
        background-color: {CARD};
        border-radius: 8px;
        border: 1px solid #1C2A40;
    }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
        background-color: {ACCENT2};
        color: {TEXT};
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {ACCENT};
        color: {BG};
    }}

    /* Inputs / selects / sliders */
    div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
        background-color: {CARD} !important;
        color: {TEXT} !important;
        border: 1px solid #1C2A40 !important;
    }}
    div[data-testid="stSlider"] > div > div > div > div {{
        background-color: {ACCENT} !important;
    }}

    /* Info / warning / success boxes */
    div[data-testid="stAlert"] {{
        background-color: {CARD};
        border-radius: 8px;
        border-left: 4px solid {ACCENT2};
    }}

    /* Radio nav in sidebar look like tabs */
    div[role="radiogroup"] label {{
        background-color: transparent;
        border-radius: 6px;
        padding: 4px 8px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Consistent color sequence for Plotly charts across the whole app
PLOTLY_COLORWAY = [ACCENT, ACCENT2, "#5EEAD4", "#7C9CBF", "#3B82F6", "#0EA5E9"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font_color=TEXT,
    colorway=PLOTLY_COLORWAY,
)


def style_fig(fig):
    """Apply the dashboard's dark theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#1C2A40", zerolinecolor="#1C2A40")
    fig.update_yaxes(gridcolor="#1C2A40", zerolinecolor="#1C2A40")
    return fig


# ============================================================
# DATA / MODEL LOADING (cached so the app stays fast)
# ============================================================
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_pickle(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

df = load_csv(DATA_PATH)
cluster_profile = load_csv(CLUSTER_PROFILE_PATH)
cluster_names = load_csv(CLUSTER_NAMES_PATH)
kmeans_eval = load_csv(KMEANS_EVAL_PATH)
model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH)
pipeline_raw = load_pickle(PIPELINE_PATH)

# customer_segmentation_pipeline.pkl is a DICT bundling all candidate models,
# not a single object with .predict(). Unpack the pieces we need for KMeans.
pipeline_features = None
if isinstance(pipeline_raw, dict):
    model = pipeline_raw.get("KMeans Model", model)
    scaler = pipeline_raw.get("Scaler", scaler)
    pipeline_features = pipeline_raw.get("Selected Features")
elif pipeline_raw is not None and hasattr(pipeline_raw, "predict"):
    # it really was a plain estimator/pipeline object
    model = pipeline_raw

predict_features = pipeline_features if pipeline_features else FEATURES

if df is None:
    st.error(f"Could not find '{DATA_PATH}'. Update DATA_PATH at the top of app.py to point at your segmented customer CSV.")
    st.stop()

# Attach human-readable segment names if the mapping file exists
SEGMENT_COL = "Cluster"
if cluster_names is not None and "Cluster" in df.columns and "Cluster" in cluster_names.columns:
    name_col = [c for c in cluster_names.columns if c != "Cluster"][0]
    df = df.merge(cluster_names, on="Cluster", how="left")
    SEGMENT_COL = name_col
elif "Cluster" not in df.columns:
    st.warning("No 'Cluster' column found in the dataset — segment-based views will be limited.")

available_features = [f for f in FEATURES if f in df.columns]

# ============================================================
# SIDEBAR — Navigation
# ============================================================
st.sidebar.title("📊 Segmentation")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "👥 Segment Explorer",
        "📈 Feature Analysis",
        "🔍 Customer Search",
        "💡 Business Recommendations",
        "🧠 Model Info",
        "🔮 Predict New Customer",
    ],
)

st.sidebar.markdown("---")


# ============================================================
# SIDEBAR — Shared filters
# ============================================================
def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("Filters")
    filtered = data.copy()

    if SEGMENT_COL in data.columns:
        segments = sorted(data[SEGMENT_COL].dropna().unique().tolist(), key=str)
        selected = st.sidebar.multiselect("Segment(s)", segments, default=segments)
        filtered = filtered[filtered[SEGMENT_COL].isin(selected)]

    if "Customer_Age" in data.columns:
        lo, hi = int(data["Customer_Age"].min()), int(data["Customer_Age"].max())
        rng = st.sidebar.slider("Age range", lo, hi, (lo, hi))
        filtered = filtered[filtered["Customer_Age"].between(*rng)]

    if "Income" in data.columns:
        lo, hi = float(data["Income"].min()), float(data["Income"].max())
        rng = st.sidebar.slider("Income range", lo, hi, (lo, hi))
        filtered = filtered[filtered["Income"].between(*rng)]

    return filtered


filtered_df = apply_filters(df)

st.sidebar.markdown("---")
st.sidebar.caption("Customer Segmentation — Module 6 & 8 outputs")


def download_button(data: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🏠 Overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(df):,}")
    k2.metric("Segments", df[SEGMENT_COL].nunique() if SEGMENT_COL in df.columns else "—")
    if "Income" in df.columns:
        k3.metric("Avg Income", f"${df['Income'].mean():,.0f}")
    if "Total_Spending" in df.columns:
        k4.metric("Avg Spending", f"${df['Total_Spending'].mean():,.0f}")

    st.markdown("### Segment Size Distribution")
    if SEGMENT_COL in df.columns:
        counts = df[SEGMENT_COL].value_counts().reset_index()
        counts.columns = [SEGMENT_COL, "Count"]
        fig = px.pie(counts, names=SEGMENT_COL, values="Count", hole=0.4)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown("### Data Preview (filtered)")
    st.dataframe(filtered_df.head(50), use_container_width=True)
    download_button(filtered_df, "⬇️ Download filtered data (CSV)", "filtered_customers.csv")


# ============================================================
# PAGE 2 — SEGMENT EXPLORER
# ============================================================
elif page == "👥 Segment Explorer":
    st.title("👥 Segment Explorer")

    if SEGMENT_COL not in df.columns:
        st.warning("No segment column available.")
    else:
        chosen = st.selectbox("Select a segment to inspect", sorted(df[SEGMENT_COL].dropna().unique().tolist(), key=str))
        seg_df = df[df[SEGMENT_COL] == chosen]

        c1, c2, c3 = st.columns(3)
        c1.metric("Customers in segment", f"{len(seg_df):,}")
        if "Income" in seg_df.columns:
            c2.metric("Avg Income", f"${seg_df['Income'].mean():,.0f}")
        if "Total_Spending" in seg_df.columns:
            c3.metric("Avg Spending", f"${seg_df['Total_Spending'].mean():,.0f}")

        if cluster_profile is not None:
            st.markdown("### Segment Profile (avg values)")
            st.dataframe(cluster_profile, use_container_width=True)

        if available_features:
            st.markdown("### Feature Averages — This Segment vs Overall")
            comp = pd.DataFrame({
                "Feature": available_features,
                "This Segment": [seg_df[f].mean() for f in available_features],
                "Overall": [df[f].mean() for f in available_features],
            })
            fig = px.bar(comp, x="Feature", y=["This Segment", "Overall"], barmode="group")
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("### Customers in this segment")
        st.dataframe(seg_df, use_container_width=True)
        download_button(seg_df, "⬇️ Download this segment (CSV)", f"segment_{chosen}.csv")


# ============================================================
# PAGE 3 — FEATURE ANALYSIS
# ============================================================
elif page == "📈 Feature Analysis":
    st.title("📈 Feature Analysis")

    if not available_features:
        st.warning("No known engineered features found in the dataset.")
    else:
        feature = st.selectbox("Choose a feature", available_features)
        chart_type = st.radio("Chart type", ["Box plot by segment", "Histogram", "Scatter vs Income"], horizontal=True)

        if chart_type == "Box plot by segment" and SEGMENT_COL in filtered_df.columns:
            fig = px.box(filtered_df, x=SEGMENT_COL, y=feature, color=SEGMENT_COL)
        elif chart_type == "Histogram":
            fig = px.histogram(filtered_df, x=feature, color=SEGMENT_COL if SEGMENT_COL in filtered_df.columns else None)
        else:
            fig = px.scatter(
                filtered_df, x="Income", y=feature,
                color=SEGMENT_COL if SEGMENT_COL in filtered_df.columns else None,
            )
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.markdown("### Summary statistics")
        st.dataframe(filtered_df[[feature] + ([SEGMENT_COL] if SEGMENT_COL in filtered_df.columns else [])]
                     .groupby(SEGMENT_COL).describe() if SEGMENT_COL in filtered_df.columns
                     else filtered_df[[feature]].describe(), use_container_width=True)


# ============================================================
# PAGE 4 — CUSTOMER SEARCH
# ============================================================
elif page == "🔍 Customer Search":
    st.title("🔍 Customer Search")

    id_cols = [c for c in df.columns if "id" in c.lower()]
    search_col = st.selectbox("Search by column", id_cols if id_cols else df.columns.tolist())
    query = st.text_input(f"Enter value for '{search_col}'")

    if query:
        try:
            result = df[df[search_col].astype(str).str.contains(query, case=False, na=False)]
        except Exception:
            result = pd.DataFrame()
        st.markdown(f"**{len(result)} match(es) found**")
        st.dataframe(result, use_container_width=True)
        if not result.empty:
            download_button(result, "⬇️ Download search results (CSV)", "search_results.csv")
    else:
        st.info("Enter a value above to search the customer base.")
        st.dataframe(df.head(20), use_container_width=True)


# ============================================================
# PAGE 5 — BUSINESS RECOMMENDATIONS
# ============================================================
elif page == "💡 Business Recommendations":
    st.title("💡 Business Recommendations")
    st.caption("Sourced from Module 8 — Executive Summary & Full Business Report")

    if SEGMENT_COL in df.columns:
        chosen = st.selectbox("View recommendation for segment", sorted(df[SEGMENT_COL].dropna().unique().tolist(), key=str))
        st.markdown(f"### {chosen}")
        st.info(
            "Paste the specific recommendation text for this segment here, "
            "or load it dynamically from your Module 8 report/notebook output."
        )
    else:
        st.write("Add your Module 8 business recommendation text/content here.")

    st.markdown("---")
    st.markdown(
        "📄 Full reports available in your project folder: `Executive_Summary.pdf`, "
        "`Full_Business_Report.pdf`, `Module8_Business_Insights.ipynb`"
    )


# ============================================================
# PAGE 6 — MODEL INFO
# ============================================================
elif page == "🧠 Model Info":
    st.title("🧠 Model Info")

    c1, c2 = st.columns(2)
    c1.metric("Algorithm", "KMeans")
    c2.metric("Number of Segments", df[SEGMENT_COL].nunique() if SEGMENT_COL in df.columns else "—")

    if kmeans_eval is not None:
        st.markdown("### Evaluation Metrics")
        st.dataframe(kmeans_eval, use_container_width=True)

    st.markdown("### Model artifact status")
    st.write(f"- KMeans model loaded: {'✅' if model is not None else '❌ (check MODEL_PATH)'}")
    st.write(f"- Scaler loaded: {'✅' if scaler is not None else '❌ (check SCALER_PATH)'}")
    st.write(f"- Pipeline bundle loaded: {'✅' if pipeline_raw is not None else '❌ (check PIPELINE_PATH)'}")
    if isinstance(pipeline_raw, dict):
        st.caption(f"Bundle contains: {', '.join(pipeline_raw.keys())}")


# ============================================================
# PAGE 7 — PREDICT NEW CUSTOMER
# ============================================================
elif page == "🔮 Predict New Customer":
    st.title("🔮 Predict New Customer Segment")

    if model is None:
        st.warning("No KMeans model found. Update MODEL_PATH or PIPELINE_PATH at the top of app.py.")
    else:
        st.write("Enter customer details:")
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(predict_features):
            default = float(df[feat].mean()) if feat in df.columns else 0.0
            input_data[feat] = cols[i % 3].number_input(feat, value=default)

        if st.button("Predict Segment"):
            # Build the input row in the EXACT column order the scaler/model were fit on
            input_df = pd.DataFrame([[input_data[f] for f in predict_features]], columns=predict_features)
            try:
                scaled = scaler.transform(input_df) if scaler is not None else input_df.values
                pred = model.predict(scaled)
                cluster_id = pred[0]
                label = cluster_id
                if cluster_names is not None and "Cluster" in cluster_names.columns:
                    label_col = [c for c in cluster_names.columns if c != "Cluster"][0]
                    match = cluster_names[cluster_names["Cluster"] == cluster_id]
                    if not match.empty:
                        label = match.iloc[0][label_col]
                st.success(f"Predicted segment: **{label}**")
            except Exception as e:
                st.error(f"Prediction failed — check that input columns match training features. Error: {e}")
