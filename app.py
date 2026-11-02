
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="📊", layout="wide")
st.title("Bank Marketing — Classification Model Demo")
st.write(
    "Upload labelled test data, select a trained classifier, and inspect its "
    "evaluation metrics and confusion matrix."
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "KNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "support_vector_machine.joblib",
}

selected_model = st.selectbox("Select a classification model", list(MODEL_PATHS))
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

st.caption("A sample held-out test dataset is provided as test_data.csv in the repository.")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded data preview")
    st.dataframe(data.head())

    if "y" not in data.columns:
        st.error("The uploaded CSV must include the target column 'y' (yes/no) to calculate evaluation metrics.")
        st.stop()

    if not MODEL_PATHS[selected_model].exists():
        st.error(f"Model file not found: {MODEL_PATHS[selected_model]}")
        st.stop()

    model = joblib.load(MODEL_PATHS[selected_model])
    X_eval = data.drop(columns=["y"])
    y_true_raw = data["y"].astype(str).str.strip().str.lower()
    mapping = {"no": 0, "yes": 1}
    y_true = y_true_raw.map(mapping)

    if y_true.isna().any():
        st.error("Target column 'y' must contain only 'yes' and 'no'.")
        st.stop()

    try:
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]
    except Exception as exc:
        st.error(f"Could not evaluate this file. Check that its columns match the training features. Details: {exc}")
        st.stop()

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.subheader(f"Evaluation — {selected_model}")
    cols = st.columns(3)
    metrics = [
        ("Accuracy", accuracy), ("AUC", auc), ("Precision", precision),
        ("Recall", recall), ("F1 Score", f1), ("MCC", mcc)
    ]
    for i, (label, value) in enumerate(metrics):
        cols[i % 3].metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1],
        display_labels=["No", "Yes"], values_format="d", ax=ax
    )
    st.pyplot(fig)

    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["No subscription", "Yes subscription"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4))

    result_download = data.copy()
    result_download["actual_y_numeric"] = y_true
    result_download["predicted_y_numeric"] = y_pred
    result_download["predicted_probability_yes"] = y_prob
    st.download_button(
        "Download predictions CSV",
        result_download.to_csv(index=False).encode("utf-8"),
        file_name="bank_marketing_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Upload test_data.csv (or another labelled test CSV) to display model results.")
