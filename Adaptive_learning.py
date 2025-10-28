#!/usr/bin/env python
# coding: utf-8

# In[2]:


# streamlit_diabetes_adaptive_app.py
"""

Adaptive diabetes-risk classifier demonstrating:
 - Data tailoring (feature engineering)
 - Clustering (KMeans phenotypes) + prediction
 - Model adaptation to cohort/distribution shift (KS test + retrain)

Features:
 - Upload your own CSV or attempt to load /content/diabetes_1.csv
 - Step-through UI to run Baseline, Tailoring, Clustering, and Adaptation
 - Displays evaluation reports, ROC-AUC, confusion matrices, and cluster visualizations
 - Fields for Student ID and Full Name (printed on report)

Usage:
    streamlit run streamlit_diabetes_adaptive_app.py

Make sure you have the required packages installed:
    pip install streamlit scikit-learn pandas numpy matplotlib scipy

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.cluster import KMeans

from scipy.stats import ks_2samp

# -------------------- Configuration --------------------
RANDOM_STATE = 42
N_CLUSTERS_DEFAULT = 4
CV_SPLITS = 5
SCORING = "f1"
EPS = 1e-6
DRIFT_PVAL_THRESHOLD = 0.01

st.set_page_config(page_title="CN7050 — Adaptive Diabetes Classifier", layout="wide")

# -------------------- Helper functions --------------------

def evaluate(model, X, y, title="Evaluation"):
    """Return classification report, roc-auc, confusion matrix and probabilities."""
    y_pred = model.predict(X)
    # some sklearn estimators don't have predict_proba (but ours will)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    report_text = classification_report(y, y_pred, digits=3, output_dict=False)
    report_dict = None
    try:
        report_dict = classification_report(y, y_pred, digits=3, output_dict=True)
    except Exception:
        report_dict = None
    auc = roc_auc_score(y, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y, y_pred)
    return {
        "report_text": report_text,
        "report_dict": report_dict,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def ks_drift(old_df, new_df, cols, p_thresh=DRIFT_PVAL_THRESHOLD):
    """Lightweight KS-test drift checker. Returns True if any column shows drift."""
    for c in cols:
        if c in old_df.columns and c in new_df.columns:
            a = pd.to_numeric(old_df[c], errors="coerce")
            b = pd.to_numeric(new_df[c], errors="coerce")
            a = a.replace([np.inf, -np.inf], np.nan).dropna()
            b = b.replace([np.inf, -np.inf], np.nan).dropna()
            if len(a) > 30 and len(b) > 30:
                _, p = ks_2samp(a, b)
                if p < p_thresh:
                    return True
    return False


# Utility to add cluster id given fitted scaler+km
def add_cluster_feature(X, fitted_scaler, fitted_kmeans):
    Z = fitted_scaler.transform(X)
    clusters = fitted_kmeans.predict(Z)
    Xc = X.copy()
    Xc["cluster_id"] = clusters.astype("float64")
    return Xc


# -------------------- Streamlit UI --------------------
st.title("CN7050 — Adaptive Diabetes Classifier (Lab 3)")

with st.sidebar:
    

    st.markdown("---")
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload diabetes CSV (must contain an 'Outcome' column)", type=["csv"]) 
    st.markdown("If you are running in Colab, a default path '/content/diabetes_1.csv' will be attempted.")

    st.markdown("---")
    st.header("Hyperparameters")
    n_clusters = st.number_input("KMeans clusters (phenotypes)", min_value=2, max_value=10, value=N_CLUSTERS_DEFAULT)
    test_size = st.slider("Test size (for baseline/tailoring)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random state", value=RANDOM_STATE)
    st.markdown("Grid search penalties and C values are fixed to: penalties=['l1','l2'], C=[0.01,0.1,1,3,10]")
    run_baseline = st.checkbox("Run Baseline (LogisticRegression)", value=True)
    run_tailoring = st.checkbox("Run Data Tailoring (engineered features)", value=True)
    run_clustering = st.checkbox("Run Clustering + Prediction", value=True)
    run_adaptation = st.checkbox("Run Model Adaptation (cohort shift)", value=True)

# Main area
col1, col2 = st.columns([2, 1])

# Load dataset
@st.cache_data
def load_csv_from_path(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        return None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Loaded uploaded CSV")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        df = None
else:
    # try default path
    df = load_csv_from_path("/content/diabetes_1.csv")
    if df is not None:
        st.info("Loaded dataset from /content/diabetes_1.csv")
    else:
        st.warning("No CSV uploaded and default path not found. Please upload a CSV with the same schema as Pima dataset (Outcome target).")

if df is not None:
    st.subheader("Dataset preview")
    st.dataframe(df.head())
    if "Outcome" not in df.columns:
        st.error("The dataset must contain an 'Outcome' column (0/1). Aborting analysis.")
    else:
        # Prepare X and y
        X_full = df.drop(columns=["Outcome"]).copy()
        y_full = df["Outcome"].astype(int).copy()
        numeric_cols = X_full.columns.tolist()

        # Show missing values
        with col2:
            st.write("**Missing values per column**")
            st.table(df.isnull().sum())

        # ------------ Baseline ------------
        if run_baseline:
            st.header("1) Baseline: StandardScaler -> LogisticRegression")
            # Train/test split
            X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
                X_full, y_full, test_size=test_size, stratify=y_full, random_state=random_state
            )
            cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)
            param_grid = {"clf__penalty": ["l1", "l2"], "clf__C": [0.01, 0.1, 1, 3, 10]}

            pipe_baseline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state))
            ])

            gs_base = GridSearchCV(
                estimator=pipe_baseline,
                param_grid=param_grid,
                cv=cv,
                scoring=SCORING,
                n_jobs=-1
            )
            with st.spinner("Fitting baseline... this can take a moment"):
                gs_base.fit(X_train_base, y_train_base)
            baseline_model = gs_base.best_estimator_
            st.write("Baseline best params:", gs_base.best_params_)
            res_base = evaluate(baseline_model, X_test_base, y_test_base, title="Baseline")
            st.text(res_base["report_text"])
            st.write("ROC-AUC:", round(res_base["roc_auc"], 3) if res_base["roc_auc"] is not None else "N/A")
            st.write("Confusion matrix:")
            st.write(res_base["confusion_matrix"])

            # ROC curve
            if res_base["y_proba"] is not None:
                fpr, tpr, _ = roc_curve(y_test_base, res_base["y_proba"])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr)
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title("Baseline ROC")
                st.pyplot(fig)

        # ------------ Data Tailoring ------------
        if run_tailoring:
            st.header("2) Data Tailoring: engineered features")
            X_tailored = X_full.copy()
            if {"Insulin", "Glucose"}.issubset(X_tailored.columns):
                X_tailored["Insulin_over_Glucose"] = X_tailored["Insulin"] / (X_tailored["Glucose"] + EPS)
            if {"BMI", "Glucose"}.issubset(X_tailored.columns):
                X_tailored["BMI_x_Glucose"] = X_tailored["BMI"] * X_tailored["Glucose"]

            X_train_tailored, X_test_tailored, y_train_tailored, y_test_tailored = train_test_split(
                X_tailored, y_full, test_size=test_size, stratify=y_full, random_state=random_state
            )

            pipe_tailored = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state))
            ])
            gs_tailored = GridSearchCV(
                estimator=pipe_tailored,
                param_grid=param_grid,
                cv=cv,
                scoring=SCORING,
                n_jobs=-1
            )
            with st.spinner("Fitting tailored model..."):
                gs_tailored.fit(X_train_tailored, y_train_tailored)
            tailored_model = gs_tailored.best_estimator_
            st.write("Data Tailoring best params:", gs_tailored.best_params_)
            res_tail = evaluate(tailored_model, X_test_tailored, y_test_tailored, title="Tailored")
            st.text(res_tail["report_text"])
            st.write("ROC-AUC:", round(res_tail["roc_auc"], 3) if res_tail["roc_auc"] is not None else "N/A")
            st.write("Confusion matrix:")
            st.write(res_tail["confusion_matrix"])

        # ------------ Clustering + Prediction ------------
        if run_clustering and run_tailoring:
            st.header("3) Clustering + Prediction (KMeans phenotype added)")
            # Fit scaler and kmeans on TRAIN of tailored
            scaler_for_kmeans = StandardScaler().fit(X_train_tailored)
            Z_train_tailored = scaler_for_kmeans.transform(X_train_tailored)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(Z_train_tailored)

            X_train_clustered = add_cluster_feature(X_train_tailored, scaler_for_kmeans, kmeans)
            X_test_clustered = add_cluster_feature(X_test_tailored, scaler_for_kmeans, kmeans)

            pipe_clustered = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state))
            ])
            gs_clustered = GridSearchCV(
                estimator=pipe_clustered,
                param_grid=param_grid,
                cv=cv,
                scoring=SCORING,
                n_jobs=-1
            )
            with st.spinner("Fitting clustered model..."):
                gs_clustered.fit(X_train_clustered, y_train_tailored)
            clustered_model = gs_clustered.best_estimator_
            st.write("Clustering+Prediction best params:", gs_clustered.best_params_)
            res_cluster = evaluate(clustered_model, X_test_clustered, y_test_tailored, title="Clustered")
            st.text(res_cluster["report_text"])
            st.write("ROC-AUC:", round(res_cluster["roc_auc"], 3) if res_cluster["roc_auc"] is not None else "N/A")
            st.write("Confusion matrix:")
            st.write(res_cluster["confusion_matrix"])

            # Visualize cluster distribution across Outcome
            with st.expander("Show cluster -> outcome cross-tab"):
                ct = pd.crosstab(X_train_clustered["cluster_id"], y_train_tailored)
                st.write(ct)

            # Plot cluster centers (in first two PCA-ish dims if many features)
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                Z = scaler_for_kmeans.transform(X_tailored)
                Zp = pca.fit_transform(Z)
                clusters_all = kmeans.predict(Z)
                fig, ax = plt.subplots()
                scatter = ax.scatter(Zp[:, 0], Zp[:, 1], c=clusters_all)
                ax.set_title("KMeans clusters (PCA projection)")
                st.pyplot(fig)
            except Exception:
                st.write("Cluster visualization not available (PCA failed)")

        # ------------ Model Adaptation (cohort shift) ------------
        if run_adaptation and run_tailoring:
            st.header("4) Model Adaptation: YoungAdults -> OlderAdults cohort shift")
            # create cohorts by Age if available
            if "Age" in X_tailored.columns:
                mask_young = X_tailored["Age"] < 40
            else:
                st.warning("'Age' column not found — falling back to splitting by median Glucose")
                mask_young = X_tailored["Glucose"] < X_tailored["Glucose"].median()

            X_cohort_young = X_tailored[mask_young].copy()
            y_cohort_young = y_full[mask_young].copy()
            X_cohort_older = X_tailored[~mask_young].copy()
            y_cohort_older = y_full[~mask_young].copy()

            st.write(f"Young cohort size: {len(X_cohort_young)}, Older cohort size: {len(X_cohort_older)}")

            # Train initial model on Young only (include clustering)
            if len(X_cohort_young) < 40 or len(X_cohort_older) < 40:
                st.warning("Cohort sizes are small; KS test may be unreliable. Proceeding anyway.")

            X_train_initial, X_valid_initial, y_train_initial, y_valid_initial = train_test_split(
                X_cohort_young, y_cohort_young, test_size=0.2, stratify=y_cohort_young, random_state=random_state
            )

            # clustering within initial cohort
            scaler_initial_for_kmeans = StandardScaler().fit(X_train_initial)
            Z_train_initial = scaler_initial_for_kmeans.transform(X_train_initial)
            kmeans_initial = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(Z_train_initial)

            def add_cluster_initial(X):
                Z = scaler_initial_for_kmeans.transform(X)
                cl = kmeans_initial.predict(Z)
                Xc = X.copy()
                Xc["cluster_id"] = cl.astype("float64")
                return Xc

            X_train_initial_c = add_cluster_initial(X_train_initial)
            X_valid_initial_c = add_cluster_initial(X_valid_initial)

            pipe_initial = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state))
            ])
            gs_initial = GridSearchCV(pipe_initial, param_grid=param_grid, cv=cv, scoring=SCORING, n_jobs=-1)
            with st.spinner("Fitting initial cohort model..."):
                gs_initial.fit(X_train_initial_c, y_train_initial)
            initial_model = gs_initial.best_estimator_
            st.write("Initial Cohort best params:", gs_initial.best_params_)
            res_init = evaluate(initial_model, X_valid_initial_c, y_valid_initial, title="Initial Cohort (Young)")
            st.text(res_init["report_text"])
            st.write("ROC-AUC:", round(res_init["roc_auc"], 3) if res_init["roc_auc"] is not None else "N/A")

            # Drift detection vs older cohort
            drift_detected = ks_drift(X_train_initial, X_cohort_older, cols=list(set(X_train_initial.columns) & set(X_cohort_older.columns)))
            st.write("Distribution drift vs Initial Cohort detected?", drift_detected)

            # Adaptation: re-learn clusters on combined and re-train
            X_combined = pd.concat([X_train_initial, X_cohort_older], axis=0)
            y_combined = pd.concat([y_train_initial, y_cohort_older], axis=0)

            scaler_combined_for_kmeans = StandardScaler().fit(X_combined)
            Z_combined = scaler_combined_for_kmeans.transform(X_combined)
            kmeans_combined = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(Z_combined)

            def add_cluster_combined(X):
                Z = scaler_combined_for_kmeans.transform(X)
                cid = kmeans_combined.predict(Z)
                Xc = X.copy()
                Xc["cluster_id"] = cid.astype("float64")
                return Xc

            X_combined_c = add_cluster_combined(X_combined)
            X_older_c = add_cluster_combined(X_cohort_older)

            pipe_adapted = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state))
            ])
            gs_adapted = GridSearchCV(pipe_adapted, param_grid=param_grid, cv=cv, scoring=SCORING, n_jobs=-1)
            with st.spinner("Fitting adapted model on combined cohorts..."):
                gs_adapted.fit(X_combined_c, y_combined)
            adapted_model = gs_adapted.best_estimator_
            st.write("Adapted Model (Combined Cohorts) best params:", gs_adapted.best_params_)
            res_adapt = evaluate(adapted_model, X_older_c, y_cohort_older, title="After Adaptation (OlderAdults)")
            st.text(res_adapt["report_text"])
            st.write("ROC-AUC:", round(res_adapt["roc_auc"], 3) if res_adapt["roc_auc"] is not None else "N/A")
            st.write("Confusion matrix (adapted on OlderAdults):")
            st.write(res_adapt["confusion_matrix"])

            # Show comparison table
            st.subheader("Summary comparison (key metrics)")
            summary = {
                "model": ["Initial_Young", "Adapted_on_combined"],
                "roc_auc": [res_init["roc_auc"], res_adapt["roc_auc"]],
                "accuracy": [
                    np.mean(res_init["y_pred"] == y_valid_initial.values) if res_init["y_pred"].shape[0] == y_valid_initial.shape[0] else None,
                    np.mean(res_adapt["y_pred"] == y_cohort_older.values) if res_adapt["y_pred"].shape[0] == y_cohort_older.shape[0] else None,
                ]
            }
            st.table(pd.DataFrame(summary))

        # -------------------- Export / report --------------------
        st.markdown("---")
        st.header("Export")
        st.write("You can download the trained adapted model (pickle) or snapshots of results by running this app locally and adding serialization code.")
        st.success("Done — adjust options on the left and re-run steps interactively.")

else:
    st.info("Waiting for a CSV to be uploaded...")

# footer
st.markdown("---")
st.caption(f"CN7050 Lab 3 app — Student: {student_name} ({student_id})")


# In[ ]:




