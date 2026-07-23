"""
☄️ NEO Hazard Predictor — Streamlit App
----------------------------------------
A friendly front-end for the ANN model trained in the "NEO Hazard Prediction"
notebook. Drop this file next to the `models_ann/` folder that the notebook
saves (Step 13) and run:

    streamlit run app.py

If `models_ann/` isn't there yet, the app still opens and tells you exactly
what to do — it won't just crash on you.
"""

import os
import json
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Keras / joblib are only needed if the real model is present, so we import
# them lazily inside the loader — keeps the app from breaking if TensorFlow
# isn't installed on a machine that only wants to browse the dashboard.


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="NEO Hazard Predictor",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models_ann"

# A couple of small style tweaks so the default Streamlit look feels less
# "default demo app" and a bit more considered.
st.markdown(
    """
    <style>
    .main > div { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
        border: 1px solid rgba(120, 120, 120, 0.15);
    }
    .risk-banner {
        padding: 18px 22px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .risk-safe { background-color: rgba(46, 204, 113, 0.15); color: #1e8449; border: 1px solid rgba(46,204,113,0.4);}
    .risk-hazard { background-color: rgba(231, 76, 60, 0.15); color: #c0392b; border: 1px solid rgba(231,76,60,0.4);}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Artifact loading (cached so we don't reload the model on every click)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Loads the trained model, scaler, feature list and metadata saved by
    Step 13 of the notebook. Returns None for anything that's missing
    instead of throwing, so the app can degrade gracefully.
    """
    model, scaler, feature_names, metadata = None, None, None, None

    meta_path = os.path.join(MODEL_DIR, "ann_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    feat_path = os.path.join(MODEL_DIR, "feature_names.json")
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            feature_names = json.load(f)

    model_path = os.path.join(MODEL_DIR, "ann_model.keras")
    scaler_path = os.path.join(MODEL_DIR, "scaler_ann.joblib")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        import joblib
        from tensorflow import keras

        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

    return model, scaler, feature_names, metadata


model, scaler, feature_names, metadata = load_artifacts()
artifacts_ready = model is not None and scaler is not None and feature_names is not None

# Fall back to the numbers baked into the notebook so the dashboard still
# looks alive even before anyone has run training. Clearly labelled as
# sample values below so nobody mistakes them for a live result.
DEMO_METRICS = {"accuracy": 0.90, "precision": 0.55, "recall": 0.85, "f1_score": 0.67, "roc_auc": 0.95}
metrics = metadata["metrics"] if metadata else DEMO_METRICS
using_demo_metrics = metadata is None

# Baseline from the notebook's Step 12 comparison, kept fixed for reference.
RF_METRICS = {"Accuracy": 0.8444, "Precision": 0.3610, "Recall": 0.7787, "F1 Score": 0.4933, "ROC-AUC": 0.9183}




# --------------------------------------------------------------------------
# Feature engineering — mirrors Step 2 of the notebook exactly
# --------------------------------------------------------------------------
def engineer_features(row: pd.DataFrame) -> pd.DataFrame:
    row = row.copy()
    row["Avg_diameter"] = (row["est_diameter_min"] + row["est_diameter_max"]) / 2
    row["Diameter_range"] = row["est_diameter_max"] - row["est_diameter_min"]
    row["Velocity_distance_ratio"] = row["relative_velocity"] / row["miss_distance"]
    row["log_velocity"] = np.log1p(row["relative_velocity"])
    row["log_miss_distance"] = np.log1p(row["miss_distance"])
    return row


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file=None):
    """
    Loads sprint1.csv from the app's working directory, or from an uploaded
    file if the local copy isn't found. Applies the same cleaning (Step 2)
    and feature engineering as the notebook, so the EDA tab reflects exactly
    what the model was trained on. Returns None if no data is available.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("sprint1.csv"):
        df = pd.read_csv("sprint1.csv")
    else:
        return None

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["absolute_magnitude"] = df["absolute_magnitude"].fillna(df["absolute_magnitude"].mean())
    df["hazardous"] = df["hazardous"].fillna(0)

    df = engineer_features(df)
    return df


def predict_hazard(input_dict: dict) -> dict:
    row = pd.DataFrame([input_dict])
    row = engineer_features(row)
    row = row[feature_names]  # keep training column order

    row_scaled = scaler.transform(row)
    prob = float(model.predict(row_scaled, verbose=0)[0][0])
    threshold = metadata["best_threshold"] if metadata else 0.5
    pred = int(prob >= threshold)

    return {
        "prediction": pred,
        "probability": prob,
        "threshold": threshold,
        "risk_label": "HAZARDOUS" if pred == 1 else "NON-HAZARDOUS",
    }


def gauge_chart(probability: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Hazard Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c0392b" if probability >= threshold else "#1e8449"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(46,204,113,0.25)"},
                    {"range": [threshold * 100, 100], "color": "rgba(231,76,60,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# --------------------------------------------------------------------------
# Sidebar — navigation + always-visible performance snapshot
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ☄️ NEO Hazard Predictor")
    st.caption("ANN model for classifying hazardous near-Earth objects")

    page = st.radio(
        "Navigate",
        ["🎯 Predict", "📈 EDA", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Model snapshot")
    if using_demo_metrics:
        st.caption("⚠️ Showing sample values — run the notebook to generate `models_ann/` for live metrics.")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    c3, c4 = st.columns(2)
    c3.metric("Precision", f"{metrics['precision']:.1%}")
    c4.metric("Recall", f"{metrics['recall']:.1%}")

    st.metric("F1 Score", f"{metrics['f1_score']:.3f}")

    st.divider()
    if artifacts_ready:
        st.success("Model loaded and ready")
    else:
        st.warning("Model not found — Predict tab is disabled until `models_ann/` exists.")
    st.caption(f"Session date: {datetime.date.today()}")


# --------------------------------------------------------------------------
# Page: Predict
# --------------------------------------------------------------------------
if page == "🎯 Predict":
    st.title("🎯 Predict Asteroid Hazard")
    st.write(
        "Enter an asteroid's observed parameters below. The model runs the same "
        "feature engineering and scaling used in training, then reports a "
        "probability and a hazardous / non-hazardous call."
    )

    if not artifacts_ready:
        st.error(
            "No trained model found in `models_ann/`. Run Steps 1–13 of the notebook "
            "first (this saves the model, scaler, and metadata), then reload this app."
        )
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Asteroid parameters")
        col1, col2 = st.columns(2)

        with col1:
            est_diameter_min = st.number_input(
                "Estimated diameter — min (km)", min_value=0.0, value=0.25, step=0.01, format="%.3f"
            )
            est_diameter_max = st.number_input(
                "Estimated diameter — max (km)", min_value=0.0, value=0.56, step=0.01, format="%.3f"
            )
            relative_velocity = st.number_input(
                "Relative velocity (km/h)", min_value=0.0, value=75000.0, step=1000.0
            )

        with col2:
            miss_distance = st.number_input(
                "Miss distance (km)", min_value=1.0, value=35_000_000.0, step=100_000.0
            )
            absolute_magnitude = st.number_input(
                "Absolute magnitude (H)", min_value=0.0, value=20.5, step=0.1
            )

        submitted = st.form_submit_button("🔮 Predict hazard", use_container_width=True)

    if submitted:
        if est_diameter_max < est_diameter_min:
            st.warning("Max diameter is smaller than min diameter — double check those two values.")

        result = predict_hazard(
            {
                "est_diameter_min": est_diameter_min,
                "est_diameter_max": est_diameter_max,
                "relative_velocity": relative_velocity,
                "miss_distance": miss_distance,
                "absolute_magnitude": absolute_magnitude,
            }
        )

        st.subheader("Result")
        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            css_class = "risk-hazard" if result["prediction"] == 1 else "risk-safe"
            icon = "🚨" if result["prediction"] == 1 else "✅"
            st.markdown(
                f"""<div class="risk-banner {css_class}">{icon} {result['risk_label']}<br>
                <span style="font-weight:400; font-size:0.95rem;">
                Probability: {result['probability']*100:.1f}% &nbsp;|&nbsp;
                Decision threshold: {result['threshold']:.2f}
                </span></div>""",
                unsafe_allow_html=True,
            )
            st.caption(
                "The threshold was tuned on the validation set to optimize F1, "
                "rather than using the default 0.50 cut-off — this matters "
                "for imbalanced datasets like this one, where hazardous "
                "asteroids are rare."
            )

        with gauge_col:
            st.plotly_chart(gauge_chart(result["probability"], result["threshold"]), use_container_width=True)

        with st.expander("See engineered features sent to the model"):
            eng_row = engineer_features(
                pd.DataFrame(
                    [
                        {
                            "est_diameter_min": est_diameter_min,
                            "est_diameter_max": est_diameter_max,
                            "relative_velocity": relative_velocity,
                            "miss_distance": miss_distance,
                            "absolute_magnitude": absolute_magnitude,
                        }
                    ]
                )
            )
            st.dataframe(eng_row.T.rename(columns={0: "value"}), use_container_width=True)


# --------------------------------------------------------------------------
# Page: Model Performance
# --------------------------------------------------------------------------
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    st.write(
        "A look at the raw dataset behind the model — class balance, feature "
        "distributions, and correlations. This uses the same cleaning and "
        "feature engineering steps as the notebook (Step 2), so what you see "
        "here is exactly what the model was trained on."
    )

    df = load_dataset()

    if df is None:
        st.warning(
            "Couldn't find `sprint1.csv` in the app's folder. "
            "Upload it below to explore it here — nothing is saved anywhere, "
            "it's only used for this session."
        )
        uploaded = st.file_uploader("Upload sprint1.csv", type="csv")
        if uploaded is not None:
            df = load_dataset(uploaded_file=uploaded)

    if df is None:
        st.stop()

    # ---- Dataset overview -------------------------------------------------
    st.subheader("Dataset overview")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Rows", f"{df.shape[0]:,}")
    o2.metric("Columns", df.shape[1])
    hazardous_count = int(df["hazardous"].sum())
    o3.metric("Hazardous", f"{hazardous_count:,}")
    o4.metric("Hazardous share", f"{hazardous_count / len(df):.1%}")

    with st.expander("Preview raw rows"):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Missing values"):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values — the cleaning steps handled everything.")
        else:
            st.dataframe(missing.rename("missing count"), use_container_width=True)

    st.divider()

    # ---- Class balance ------------------------------------------------
    st.subheader("Class balance")
    st.caption("This is *why* the notebook uses class weights during training — hazardous asteroids are rare.")
    class_counts = df["hazardous"].value_counts().rename({0: "Non-Hazardous", 1: "Hazardous"})
    cb1, cb2 = st.columns([1, 1])
    with cb1:
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            color=class_counts.index,
            color_discrete_map={"Non-Hazardous": "#3498db", "Hazardous": "#e74c3c"},
            hole=0.4,
        )
        fig_pie.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
    with cb2:
        st.dataframe(
            class_counts.rename("count").to_frame().assign(share=lambda d: (d["count"] / d["count"].sum())),
            use_container_width=True,
        )
        ratio = class_counts["Non-Hazardous"] / class_counts["Hazardous"]
        st.metric("Imbalance ratio", f"{ratio:.1f} : 1")

    st.divider()

    # ---- Feature distributions --------------------------------------------
    st.subheader("Feature distributions")
    numeric_cols = [c for c in df.columns if c != "hazardous" and pd.api.types.is_numeric_dtype(df[c])]
    selected_feature = st.selectbox("Choose a feature to inspect", numeric_cols, index=0)

    fig_hist = px.histogram(
        df,
        x=selected_feature,
        color=df["hazardous"].map({0: "Non-Hazardous", 1: "Hazardous"}),
        color_discrete_map={"Non-Hazardous": "#3498db", "Hazardous": "#e74c3c"},
        barmode="overlay",
        opacity=0.6,
        nbins=50,
    )
    fig_hist.update_layout(height=420, legend_title_text="Class")
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_box = px.box(
        df,
        x=df["hazardous"].map({0: "Non-Hazardous", 1: "Hazardous"}),
        y=selected_feature,
        color=df["hazardous"].map({0: "Non-Hazardous", 1: "Hazardous"}),
        color_discrete_map={"Non-Hazardous": "#3498db", "Hazardous": "#e74c3c"},
    )
    fig_box.update_layout(height=380, showlegend=False, xaxis_title="Class")
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ---- Correlation heatmap ------------------------------------------------
    st.subheader("Feature correlations")
    st.caption("Includes the engineered features (average diameter, log-scaled velocity, etc.) alongside the originals.")
    corr = df[numeric_cols + ["hazardous"]].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(height=550)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.caption(
        "Tip: look at each feature's correlation with the `hazardous` row/column "
        "— that's a quick read on which inputs the model likely leans on most."
    )


elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    if using_demo_metrics:
        st.info(
            "These are placeholder numbers so you can see the layout. "
            "Run the notebook end-to-end to populate `models_ann/ann_metadata.json` "
            "with your real test-set results."
        )

    st.subheader("Test set metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{metrics['precision']:.1%}")
    m3.metric("Recall", f"{metrics['recall']:.1%}")
    m4.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.divider()
    st.subheader("ANN vs Random Forest baseline")
    st.caption("Comparison against the Random Forest model from an earlier sprint in the same project.")

    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ann_values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["roc_auc"]]
    rf_values = [RF_METRICS[l] for l in labels]

    fig = go.Figure()
    fig.add_bar(name="Random Forest", x=labels, y=rf_values, marker_color="#3498db")
    fig.add_bar(name="ANN (this model)", x=labels, y=ann_values, marker_color="#e74c3c")
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05], height=420, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    if metadata:
        st.divider()
        st.subheader("Architecture & training setup")
        arch = metadata.get("architecture", {})
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(
                f"""
                - **Hidden layers:** {' → '.join(str(u) for u in arch.get('layers', []))} → 1
                - **Activation:** ReLU (hidden), Sigmoid (output)
                - **Optimizer:** {arch.get('optimizer', 'Adam')}
                - **Loss:** {arch.get('loss', 'binary_crossentropy')}
                """
            )
        with a2:
            st.markdown(
                f"""
                - **Dropout rate:** {arch.get('dropout', '—')}
                - **L2 regularization:** {arch.get('l2_reg', '—')}
                - **Decision threshold:** {metadata.get('best_threshold', '—')}
                - **Input features:** {metadata.get('n_features', '—')}
                """
            )


# --------------------------------------------------------------------------
# Page: About
# --------------------------------------------------------------------------
else:
    st.title("ℹ️ About this project")
    st.write(
        """
        This app is a front-end for a Near-Earth Object (NEO) hazard classifier.
        The underlying model is an Artificial Neural Network (ANN) built with
        TensorFlow/Keras, trained to flag whether an asteroid should be
        considered potentially hazardous based on its physical and orbital
        characteristics.
        """
    )

    st.subheader("How the model was built")
    st.markdown(
        """
        1. **Data cleaning** — missing values in `absolute_magnitude` and `hazardous`
           were filled in, and an unnamed index column was dropped.
        2. **Feature engineering** — five new features were derived: average
           diameter, diameter range, velocity/distance ratio, and log-scaled
           velocity and miss distance.
        3. **Train / validation / test split** — an 80/20 split, with a further
           15% of the training set carved out for validation, all stratified
           to preserve the hazardous/non-hazardous ratio.
        4. **Class balancing** — since hazardous asteroids are a small minority
           of the data (roughly a 9:1 imbalance), class weights were computed
           and applied during training.
        5. **Architecture** — a 3-hidden-layer network with batch normalization,
           dropout, and L2 regularization, tuned via a manual grid search over
           layer sizes, dropout rates, and learning rates.
        6. **Threshold tuning** — rather than using a default 0.5 cut-off, the
           decision threshold was optimized on the validation set to get the
           best F1 score, which matters a lot when the positive class is rare.
        7. **Evaluation** — final metrics (accuracy, precision, recall, F1,
           ROC-AUC) were computed on a held-out test set never seen during
           training or tuning.
        """
    )

    st.subheader("Why these metrics matter here")
    st.markdown(
        """
        For a hazard classifier, **recall** is arguably the most important
        number on the sidebar — missing a genuinely hazardous asteroid (a false
        negative) is a far worse outcome than a false alarm. Accuracy alone can
        be misleading on an imbalanced dataset like this one, since a model
        could score highly just by predicting "non-hazardous" every time.
        That's also why F1 score and ROC-AUC are tracked alongside it — they
        give a fuller picture of how well the model balances catching real
        threats against not crying wolf too often.
        """
    )

    st.divider()
    st.caption("Built with Streamlit · TensorFlow/Keras · scikit-learn · Plotly")