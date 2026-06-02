import os
import sys
import joblib
import io
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from PIL import Image as PILImage

# Ensure package imports (classification, agent, aiops) resolve.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

_CRUNCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _get_model_artifact_path() -> str:
    """Return the path to the churn pipeline joblib artifact."""
    candidate_paths = [
        os.path.join(_CRUNCH_ROOT, "customer_crunch", "saved_models", "churn_pipeline.joblib"),
        os.path.join(_CRUNCH_ROOT, "customer crunch", "saved_models", "churn_pipeline.joblib"),
        os.path.join(_CRUNCH_ROOT, "saved_models", "churn_pipeline.joblib"),
        os.path.join(os.getcwd(), "saved_models", "churn_pipeline.joblib"),
        os.path.join(os.getcwd(), "customer_crunch", "saved_models", "churn_pipeline.joblib"),
        os.path.join(os.getcwd(), "customer crunch", "saved_models", "churn_pipeline.joblib"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p

    # Fallback for HF Docker where CWD is usually /app
    return os.path.join("saved_models", "churn_pipeline.joblib")


def _load_pipeline(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. "
            "Expected churn_pipeline.joblib under saved_models/."
        )
    return joblib.load(model_path)


# Load once at import time so Gradio requests are fast.
MODEL_PATH = os.environ.get("CHURN_PIPELINE_PATH", _get_model_artifact_path())
PIPELINE = None
PIPELINE_LOAD_ERROR = None
try:
    PIPELINE = _load_pipeline(MODEL_PATH)
except Exception as e:  # pragma: no cover
    PIPELINE_LOAD_ERROR = e


REQUIRED_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

_SHAP_EXPLAINER = None
_SHAP_PREPROCESSOR = None
_SHAP_FEATURE_NAMES = None


def _init_shap():
    """Build SHAP explainer once from the loaded pipeline."""
    global _SHAP_EXPLAINER, _SHAP_PREPROCESSOR, _SHAP_FEATURE_NAMES
    if _SHAP_EXPLAINER is not None:
        return True, None
    if PIPELINE is None:
        return False, str(PIPELINE_LOAD_ERROR)
    try:
        artifact = PIPELINE if not isinstance(PIPELINE, dict) else PIPELINE
        pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
        preprocessor = pipeline.named_steps["preprocessor"]
        xgb_model = pipeline.named_steps["classifier"]
        num_features = [
            "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
        ]
        cat_encoder = preprocessor.named_transformers_["cat"]
        enc_cats = cat_encoder.get_feature_names_out(["Geography", "Gender"]).tolist()
        _SHAP_FEATURE_NAMES = num_features + enc_cats
        _SHAP_PREPROCESSOR = preprocessor
        _SHAP_EXPLAINER = shap.TreeExplainer(xgb_model)
        return True, None
    except Exception as ex:
        return False, str(ex)


def _compute_shap(payload: dict):
    """Return (shap_values, feature_names, base_value, churn_prob) or raise."""
    ok, err = _init_shap()
    if not ok:
        raise RuntimeError(err)
    df = pd.DataFrame([payload])
    pipeline = PIPELINE["pipeline"] if isinstance(PIPELINE, dict) else PIPELINE
    X_t = _SHAP_PREPROCESSOR.transform(df)
    sv = _SHAP_EXPLAINER.shap_values(X_t)[0]
    base = float(_SHAP_EXPLAINER.expected_value)
    prob = float(pipeline.predict_proba(df)[0][1])
    return sv, _SHAP_FEATURE_NAMES, base, prob


def _shap_waterfall_image(sv, names, base, prob):
    """Return PIL Image for the signed SHAP bar chart."""
    from PIL import Image as PILImage
    pairs = sorted(zip(sv, names), key=lambda x: abs(x[0]), reverse=True)[:10]
    vals = [p[0] for p in pairs]
    lbls = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if v >= 0 else "#2ca02c" for v in vals]
    ax.barh(range(len(vals)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(lbls[::-1], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (+  toward churn  |  −  away from churn)")
    ax.set_title(f"Top 10 Feature Impacts  (churn prob = {prob*100:.1f}%,  base = {base:.3f})")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf)


def _shap_importance_image(sv, names):
    """Return PIL Image for the |SHAP| importance chart."""
    from PIL import Image as PILImage
    pairs = sorted(zip(np.abs(sv), names), key=lambda x: x[0], reverse=True)[:10]
    vals = [p[0] for p in pairs]
    lbls = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(vals)), vals[::-1], color="#1f77b4")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(lbls[::-1], fontsize=10)
    ax.set_xlabel("|SHAP value|  —  feature importance for this prediction")
    ax.set_title("Feature Importance (this customer)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf)


# ---------------------------------------------------------------------------
# Business metrics helpers
# ---------------------------------------------------------------------------

def compute_business_metrics(clv: float, offer_cost: float):
    """Run profit curve analysis on the loaded model using the training dataset."""
    try:
        from classification.business_metrics import (
            cost_benefit_matrix, expected_maximum_profit,
            roi_of_retention, format_business_report, plot_profit_curve,
        )
    except ImportError as e:
        return None, None, f"business_metrics module not available: {e}"

    if PIPELINE is None:
        return None, None, f"Model not loaded: {PIPELINE_LOAD_ERROR}"

    # Locate the reference dataset — try canonical name first
    data_candidates = [
        os.path.join(_PKG_ROOT, "data", "customer_churn_dataset.csv"),
        os.path.join(_PKG_ROOT, "data", "raw", "customer_churn_dataset.csv"),
        os.path.join(os.getcwd(), "customer_crunch", "data", "customer_churn_dataset.csv"),
        "/app/customer_crunch/data/customer_churn_dataset.csv",
        "/app/data/customer_churn_dataset.csv",
        # legacy fallback
        os.path.join(_PKG_ROOT, "data", "raw", "Churn_Modelling kaggel.csv"),
        os.path.join(os.getcwd(), "customer_crunch", "data", "raw", "Churn_Modelling kaggel.csv"),
    ]
    data_path = next((p for p in data_candidates if os.path.exists(p)), None)
    if data_path is None:
        return None, None, "Reference dataset not found. Cannot compute profit metrics."

    try:
        df = pd.read_csv(data_path)
        pipeline = PIPELINE["pipeline"] if isinstance(PIPELINE, dict) else PIPELINE

        target = "Exited" if "Exited" in df.columns else "Churn"
        drop_cols = ["CustomerId", "Surname", "RowNumber", "customerID"]
        X = df.drop(columns=drop_cols + [target], errors="ignore")
        y = df[target].values

        y_prob = pipeline.predict_proba(X)[:, 1]

        cb         = cost_benefit_matrix(clv=float(clv), offer_cost=float(offer_cost))
        emp_result = expected_maximum_profit(y, y_prob, cb=cb)
        roi_result = roi_of_retention(y, y_prob, emp_result["optimal_threshold"], cb=cb)

        report_md = format_business_report(emp_result, roi_result)

        # Profit curve image
        fig = plot_profit_curve(
            emp_result["curve_df"],
            opt_threshold=emp_result["optimal_threshold"],
            emp=emp_result["emp"],
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        profit_img = PILImage.open(buf)

        return profit_img, report_md, ""

    except Exception as ex:
        return None, None, f"Error computing business metrics: {ex}"



def predict_customer_churn(
    gender: str,
    geography: str,
    credit_score: float,
    age: float,
    balance: float,
    num_products: float,
    has_credit_card: str,
    is_active_member: str,
    estimated_salary: float,
    tenure: float,
):
    """Predict churn risk using the business-calibrated threshold."""
    if PIPELINE_LOAD_ERROR is not None or PIPELINE is None:
        return "Model failed to load.", "0.00%", "", f"{PIPELINE_LOAD_ERROR}"

    payload = {
        "CreditScore":      float(credit_score),
        "Geography":        str(geography),
        "Gender":           str(gender),
        "Age":              float(age),
        "Tenure":           float(tenure),
        "Balance":          float(balance),
        "NumOfProducts":    int(num_products),
        "HasCrCard":        1 if has_credit_card == "Yes" else 0,
        "IsActiveMember":   1 if is_active_member == "Yes" else 0,
        "EstimatedSalary":  float(estimated_salary),
    }

    try:
        from classification.predict import predict_single_customer
        result = predict_single_customer(payload, model_path=MODEL_PATH)
        prob      = result["churn_probability"]
        risk_tier = result["risk_tier"]
        opt_t     = result["optimal_threshold"]
        b_pred    = result["business_prediction"]
        status    = result["status"]

        prob_str  = f"{prob * 100:.2f}%"
        note      = (
            f"Risk tier: {risk_tier} | "
            f"Business threshold: {opt_t:.2f} | "
            f"Business decision: {'Intervene' if b_pred else 'Monitor'}"
        )
        return status, prob_str, risk_tier, note
    except Exception as ex:
        return "Error", "0.00%", "", str(ex)


def explain_customer_shap(
    gender: str,
    geography: str,
    credit_score: float,
    age: float,
    balance: float,
    num_products: float,
    has_credit_card: str,
    is_active_member: str,
    estimated_salary: float,
    tenure: float,
):
    """Compute live SHAP explanation for a single customer."""
    payload = {
        "CreditScore": float(credit_score),
        "Geography": str(geography),
        "Gender": str(gender),
        "Age": float(age),
        "Tenure": float(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": 1 if has_credit_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "EstimatedSalary": float(estimated_salary),
    }

    try:
        sv, names, base, prob = _compute_shap(payload)
    except Exception as ex:
        empty = plt.figure()
        plt.close(empty)
        return None, None, f"SHAP error: {ex}", ""

    # Build impact table
    pairs = sorted(zip(sv, names), key=lambda x: abs(x[0]), reverse=True)[:10]
    table_md = "| Feature | SHAP Value | Direction |\n|---|---:|---:|\n"
    for n, v in pairs:
        direction = "↑ toward churn" if v > 0 else "↓ away from churn"
        table_md += f"| {n} | {v:+.4f} | {direction} |\n"

    # Plain-language summary
    top_pos = [(n, v) for n, v in pairs if v > 0][:3]
    top_neg = [(n, v) for n, v in pairs if v < 0][:3]
    summary_lines = [f"**Churn probability: {prob*100:.2f}%** (model base rate: {base*100:.2f}%)\n"]
    if top_pos:
        drivers = ", ".join(f"**{n}** (+{v:.3f})" for n, v in top_pos)
        summary_lines.append(f"- Factors **increasing** churn risk: {drivers}")
    if top_neg:
        guards = ", ".join(f"**{n}** ({v:.3f})" for n, v in top_neg)
        summary_lines.append(f"- Factors **reducing** churn risk: {guards}")
    summary = "\n".join(summary_lines)

    waterfall_buf = _shap_waterfall_image(sv, names, base, prob)
    importance_buf = _shap_importance_image(sv, names)

    return waterfall_buf, importance_buf, table_md, summary


with gr.Blocks(title="Customer Churn Prediction") as demo:
    gr.Markdown("# 📊 Customer Crunch — Churn Intelligence Platform")
    gr.Markdown(
        "Predict churn, get **SHAP explanations**, chat with the **Advisor Agent**, "
        "or run the **MLOps Agent** for drift monitoring and self-healing retrain."
    )

    advisor_agent = None
    mlops_agent = None
    if PIPELINE_LOAD_ERROR is None and PIPELINE is not None:
        from agent.advisor import ChurnAdvisorAgent
        from agent.mlops_agent import MLOpsAgent

        advisor_agent = ChurnAdvisorAgent(model_path=MODEL_PATH)
        mlops_agent = MLOpsAgent(model_path=MODEL_PATH)

    # ------------------------------------------------------------------
    # Shared customer input block (reused across Predict and SHAP tabs)
    # ------------------------------------------------------------------
    def _customer_input_components(prefix: str):
        """Return a flat list of (component, key_name) for a customer form."""
        with gr.Row():
            tenure = gr.Slider(0, 10, value=5, step=1, label="Tenure (Years)")
            estimated_salary = gr.Number(value=100000, label="Estimated Salary ($)")
        with gr.Row():
            credit_score = gr.Slider(350, 850, value=600, step=1, label="Credit Score")
            age = gr.Slider(18, 92, value=40, step=1, label="Age (Years)")
        with gr.Row():
            geography = gr.Dropdown(
                choices=["France", "Germany", "Spain"], value="Germany", label="Geography"
            )
            gender = gr.Dropdown(choices=["Male", "Female"], value="Female", label="Gender")
        with gr.Row():
            balance = gr.Number(value=75000.0, label="Balance ($)")
            num_products = gr.Slider(1, 4, value=2, step=1, label="Num of Products")
        with gr.Row():
            has_credit_card = gr.Radio(choices=["Yes", "No"], value="No", label="Has Credit Card")
            is_active_member = gr.Radio(choices=["Yes", "No"], value="No", label="Is Active Member")
        return tenure, estimated_salary, credit_score, age, geography, gender, \
               balance, num_products, has_credit_card, is_active_member

    with gr.Tabs():

        # ==================================================================
        # TAB 1 — PREDICT
        # ==================================================================
        with gr.Tab("Predict"):
            gr.Markdown(
                "Enter customer attributes to get a churn risk score. "
                "Prediction uses the **business-calibrated threshold** (optimal profit point) "
                "rather than a fixed 0.5 cut-off."
            )

            with gr.Row():
                status_out   = gr.Textbox(label="Churn Status",       interactive=False)
                prob_out     = gr.Textbox(label="Churn Probability",   interactive=False)
            with gr.Row():
                tier_out     = gr.Textbox(label="Risk Tier",           interactive=False)
                details_out  = gr.Textbox(label="Business Decision",   interactive=False)

            gr.Markdown("---")
            gr.Markdown("## Customer Attributes")
            (p_tenure, p_salary, p_credit, p_age, p_geo, p_gender,
             p_balance, p_products, p_card, p_active) = _customer_input_components("p")

            run_btn = gr.Button("Predict", variant="primary")
            run_btn.click(
                fn=predict_customer_churn,
                inputs=[
                    p_gender, p_geo, p_credit, p_age,
                    p_balance, p_products, p_card, p_active, p_salary, p_tenure,
                ],
                outputs=[status_out, prob_out, tier_out, details_out],
            )

        # ==================================================================
        # TAB 2 — SHAP EXPLAINABILITY
        # ==================================================================
        with gr.Tab("SHAP Explainability"):
            gr.Markdown(
                "## 🧠 SHAP Feature Attribution\n"
                "Live SHAP values computed from the XGBoost model for any customer profile. "
                "Red bars push toward churn; green bars push away. "
                "All values are calculated on-the-fly — nothing is hardcoded."
            )

            if PIPELINE_LOAD_ERROR is not None:
                gr.Markdown(f"⚠️ Model unavailable: {PIPELINE_LOAD_ERROR}")
            else:
                gr.Markdown("### Customer Profile")
                (s_tenure, s_salary, s_credit, s_age, s_geo, s_gender,
                 s_balance, s_products, s_card, s_active) = _customer_input_components("s")

                shap_btn = gr.Button("Generate SHAP Explanation", variant="primary")

                with gr.Row():
                    shap_waterfall = gr.Image(label="Signed Impact (red = toward churn, green = away)", type="pil")
                    shap_importance = gr.Image(label="Feature Importance (|SHAP|)", type="pil")

                shap_table = gr.Markdown(label="Impact Table")
                shap_summary = gr.Markdown(label="Plain-language Summary")

                shap_btn.click(
                    fn=explain_customer_shap,
                    inputs=[
                        s_gender, s_geo, s_credit, s_age,
                        s_balance, s_products, s_card, s_active, s_salary, s_tenure,
                    ],
                    outputs=[shap_waterfall, shap_importance, shap_table, shap_summary],
                )

        # ==================================================================
        # TAB 3 — BUSINESS METRICS
        # ==================================================================
        with gr.Tab("Business Metrics"):
            gr.Markdown(
                "## 💰 Business-Oriented Evaluation\n"
                "Addresses the research gap: *'Limited use of business-oriented metrics'* "
                "(Systematic Review, 2025). Converts model predictions into profit curves, "
                "Expected Maximum Profit (EMP), and Retention ROI — going beyond accuracy/F1.\n\n"
                "Adjust the cost assumptions below and click **Compute** to run the analysis."
            )

            if PIPELINE_LOAD_ERROR is not None:
                gr.Markdown(f"⚠️ Model unavailable: {PIPELINE_LOAD_ERROR}")
            else:
                with gr.Row():
                    bm_clv = gr.Number(
                        value=200.0,
                        label="Customer Lifetime Value — CLV ($)",
                        info="Revenue saved when a churner is successfully retained",
                    )
                    bm_offer = gr.Number(
                        value=20.0,
                        label="Retention Offer Cost ($)",
                        info="Cost of sending one retention offer (per contacted customer)",
                    )

                bm_btn = gr.Button("Compute Business Metrics", variant="primary")

                bm_profit_img = gr.Image(
                    label="Profit Curve — net profit at every classification threshold",
                    type="pil",
                )
                bm_report = gr.Markdown(label="Business Evaluation Report")
                bm_error  = gr.Textbox(label="Errors", interactive=False, visible=False)

                def _run_bm(clv, offer):
                    img, report, err = compute_business_metrics(clv, offer)
                    return img, report or "", gr.update(value=err, visible=bool(err))

                bm_btn.click(
                    fn=_run_bm,
                    inputs=[bm_clv, bm_offer],
                    outputs=[bm_profit_img, bm_report, bm_error],
                )

        # ==================================================================
        # TAB 4 — ADVISOR AGENT
        # ==================================================================
        with gr.Tab("Advisor Agent"):
            # Show LLM availability status clearly
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            hf_model = os.environ.get("HF_LLM_MODEL")
            if hf_token and hf_model:
                gr.Markdown(
                    f"🟢 **LLM mode active** — using `{hf_model}` via HuggingFace Inference API.  "
                    "Open-ended questions will be answered by the language model."
                )
            else:
                gr.Markdown(
                    "🟡 **Rule-based mode** — LLM enhancement is off.  "
                    "Set `HF_TOKEN` and `HF_LLM_MODEL` environment variables to enable it.  \n"
                    "Supported commands: `predict`, `predict age=52 geography=Germany`, `tips`, `help`."
                )

            if advisor_agent is None:
                gr.Markdown(f"⚠️ Advisor unavailable: {PIPELINE_LOAD_ERROR}")
            else:
                def advisor_chat(message, history):
                    return advisor_agent.reply(message, history)

                gr.ChatInterface(
                    fn=advisor_chat,
                    examples=[
                        "help",
                        "predict",
                        "predict age=55 tenure=1 is_active=no credit=480 geography=Spain",
                        "tips",
                    ],
                    title="Churn Advisor Agent",
                )

        # ==================================================================
        # TAB 5 — MLOPS AGENT
        # ==================================================================
        with gr.Tab("MLOps Agent"):
            gr.Markdown(
                "Monitor **data drift** (KS test) and trigger **self-healing retrain** "
                "when feature distributions shift."
            )
            if mlops_agent is None:
                gr.Markdown(f"⚠️ MLOps agent unavailable: {PIPELINE_LOAD_ERROR}")
            else:
                with gr.Row():
                    drift_threshold = gr.Slider(
                        1, 5, value=1, step=1, label="Drift features to trigger retrain"
                    )
                    alpha = gr.Slider(
                        0.01, 0.2, value=0.05, step=0.01, label="Significance (alpha)"
                    )
                simulate_drift = gr.Checkbox(
                    label="Simulate drift (shift Age +10 for demo)", value=False
                )
                dry_run = gr.Checkbox(
                    label="Dry run (report only, no retrain)", value=True
                )
                mlops_out = gr.Markdown(label="MLOps report")

                with gr.Row():
                    scan_btn = gr.Button("Run drift scan", variant="secondary")
                    cycle_btn = gr.Button("Run full MLOps cycle", variant="primary")

                def _scan_only(threshold, a, sim, dry):
                    report = mlops_agent.run_drift_scan(alpha=a, simulate_drift=sim)
                    heal = mlops_agent.run_self_heal(
                        report, drift_threshold=int(threshold), dry_run=dry
                    )
                    suffix = (
                        f"\n\n**Dry-run self-heal:** would_retrain={heal.get('would_retrain')}"
                        if dry else
                        f"\n\n**Self-heal status:** `{heal.get('status')}`"
                    )
                    return mlops_agent.format_drift_report(report) + suffix

                def _full_cycle(threshold, a, sim, dry):
                    return mlops_agent.run_full_cycle(
                        drift_threshold=int(threshold),
                        alpha=a,
                        simulate_drift=sim,
                        dry_run=dry,
                    )

                scan_btn.click(
                    _scan_only,
                    inputs=[drift_threshold, alpha, simulate_drift, dry_run],
                    outputs=mlops_out,
                )
                cycle_btn.click(
                    _full_cycle,
                    inputs=[drift_threshold, alpha, simulate_drift, dry_run],
                    outputs=mlops_out,
                )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        root_path=os.getenv("SPACE_URL_PATH", ""),
        show_error=True,
    )

