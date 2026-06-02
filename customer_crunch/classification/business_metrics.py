"""Business-oriented evaluation metrics for churn prediction.

Addresses the research gap identified in:
  "Customer Churn Prediction: A Systematic Review of Recent Advances,
   Trends, and Challenges in ML and DL" (2025)

Gap: Research heavily emphasises accuracy/F1/AUC while profit-based
evaluation is rarely considered.

Implemented metrics
-------------------
1. Expected Maximum Profit (EMP)
   Based on Verbraken et al. (2013) — the industry-standard framework
   for profit-driven churn evaluation.

2. Cost-sensitive confusion matrix
   Assigns real monetary values to TP, FP, FN, TN outcomes.

3. Profit curve
   Net profit at every classification threshold (0..1 step 0.01).

4. Optimal threshold selection
   The probability cut-off that maximises profit, not accuracy.

5. ROI of retention programme
   What % return the model-guided campaign yields over random targeting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Default cost/benefit assumptions (all values in currency units, e.g. USD)
# ---------------------------------------------------------------------------
DEFAULT_CLV = 200.0        # Customer Lifetime Value — revenue saved per TP
DEFAULT_OFFER_COST = 20.0  # Retention offer cost per contacted customer (TP + FP)
DEFAULT_FN_COST = 0.0      # Cost of missing a churner beyond lost CLV (e.g. bad-will)


def cost_benefit_matrix(
    clv: float = DEFAULT_CLV,
    offer_cost: float = DEFAULT_OFFER_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> Dict[str, float]:
    """Return a labelled cost/benefit matrix.

    Outcome mapping
    ---------------
    TP : correctly predicted churner contacted  → save CLV, pay offer_cost
    FP : non-churner contacted (wasted offer)   → pay offer_cost, gain nothing
    FN : churner missed                         → lose CLV + any fn_cost
    TN : non-churner correctly left alone       → no cost, no gain
    """
    return {
        "TP": clv - offer_cost,   # net benefit of saving a churner
        "FP": -offer_cost,        # wasted spend on non-churner
        "FN": -(clv + fn_cost),   # lost revenue from missed churner
        "TN": 0.0,                # correctly ignored — no impact
    }


def compute_profit_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cb: Dict[str, float],
) -> float:
    """Net profit when using *threshold* as the classification cut-off."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp * cb["TP"] + fp * cb["FP"] + fn * cb["FN"] + tn * cb["TN"]


def profit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cb: Optional[Dict[str, float]] = None,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute net profit at every threshold.

    Returns a DataFrame with columns:
        threshold, profit, tp, fp, fn, tn, precision, recall
    """
    if cb is None:
        cb = cost_benefit_matrix()
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        profit = tp * cb["TP"] + fp * cb["FP"] + fn * cb["FN"] + tn * cb["TN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append(
            dict(threshold=round(t, 2), profit=profit,
                 tp=tp, fp=fp, fn=fn, tn=tn,
                 precision=round(precision, 4), recall=round(recall, 4))
        )
    return pd.DataFrame(rows)


def optimal_threshold(curve_df: pd.DataFrame) -> Tuple[float, float]:
    """Return (best_threshold, max_profit) from a profit_curve DataFrame."""
    idx = curve_df["profit"].idxmax()
    row = curve_df.loc[idx]
    return float(row["threshold"]), float(row["profit"])


def expected_maximum_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cb: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute EMP — the profit at the optimal threshold.

    Returns a summary dict with:
        emp          : maximum expected profit (absolute, in currency units)
        emp_per_cust : EMP normalised by total customers
        opt_threshold: probability threshold that achieves EMP
        baseline_profit : profit if you contact everyone (threshold=0)
        random_profit   : expected profit of a random 50% campaign
        lift            : emp / random_profit  (>1 means model adds value)
        curve_df        : full profit curve DataFrame
    """
    if cb is None:
        cb = cost_benefit_matrix()

    curve_df = profit_curve(y_true, y_prob, cb)
    opt_t, emp = optimal_threshold(curve_df)

    n = len(y_true)
    n_churners = int(y_true.sum())
    churn_rate = n_churners / n if n > 0 else 0.0

    # Baseline: contact nobody — profit = FN cost for all churners
    baseline_no_contact = n_churners * cb["FN"]

    # Baseline: contact everyone (threshold = 0)
    baseline_all = compute_profit_at_threshold(y_true, y_prob, threshold=0.0, cb=cb)

    # Random campaign (contact ~50% randomly)
    random_tp = n_churners * 0.5
    random_fp = (n - n_churners) * 0.5
    random_fn = n_churners * 0.5
    random_profit = (random_tp * cb["TP"] + random_fp * cb["FP"]
                     + random_fn * cb["FN"])

    lift = emp / random_profit if random_profit != 0 else float("inf")

    return {
        "emp": round(emp, 2),
        "emp_per_customer": round(emp / n, 4) if n > 0 else 0.0,
        "optimal_threshold": opt_t,
        "baseline_contact_all": round(baseline_all, 2),
        "baseline_contact_none": round(baseline_no_contact, 2),
        "random_campaign_profit": round(random_profit, 2),
        "lift_over_random": round(lift, 4),
        "churn_rate": round(churn_rate, 4),
        "n_customers": n,
        "n_churners": n_churners,
        "curve_df": curve_df,
        "cost_benefit": cb,
    }


def roi_of_retention(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    opt_threshold: float,
    cb: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute ROI of the retention programme at the optimal threshold.

    ROI = (Net Profit) / (Total Spend) × 100%
    Total Spend = (TP + FP) × offer_cost
    """
    if cb is None:
        cb = cost_benefit_matrix()

    y_pred = (y_prob >= opt_threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    net_profit = tp * cb["TP"] + fp * cb["FP"] + fn * cb["FN"] + tn * cb["TN"]
    total_spend = (tp + fp) * abs(cb["FP"])  # offer_cost per contacted customer
    roi = (net_profit / total_spend * 100) if total_spend > 0 else 0.0

    return {
        "net_profit": round(net_profit, 2),
        "total_spend": round(total_spend, 2),
        "roi_pct": round(roi, 2),
        "customers_contacted": tp + fp,
        "churners_saved": tp,
        "churners_missed": fn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def plot_profit_curve(
    curve_df: pd.DataFrame,
    opt_threshold: float,
    emp: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot profit vs. threshold with the optimal point highlighted."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve_df["threshold"], curve_df["profit"],
            color="#1f77b4", linewidth=2, label="Profit curve")
    ax.axvline(opt_threshold, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Optimal threshold = {opt_threshold:.2f}")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.scatter([opt_threshold], [emp], color="#d62728", zorder=5, s=80)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Net profit ($)")
    ax.set_title(f"Profit Curve — EMP = ${emp:,.0f} at threshold {opt_threshold:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def format_business_report(
    emp_result: Dict[str, Any],
    roi_result: Dict[str, Any],
) -> str:
    """Return a human-readable markdown business evaluation report."""
    cb = emp_result["cost_benefit"]
    lines = [
        "## 💰 Business-Oriented Evaluation Report",
        "",
        "### Cost / Benefit Assumptions",
        f"| Outcome | Value |",
        f"|---|---:|",
        f"| TP — saved churner (CLV − offer cost) | ${cb['TP']:+.2f} |",
        f"| FP — wasted offer on non-churner | ${cb['FP']:+.2f} |",
        f"| FN — missed churner (lost CLV) | ${cb['FN']:+.2f} |",
        f"| TN — correctly ignored | ${cb['TN']:+.2f} |",
        "",
        "### Expected Maximum Profit (EMP)",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| **EMP (total)** | **${emp_result['emp']:,.2f}** |",
        f"| EMP per customer | ${emp_result['emp_per_customer']:.4f} |",
        f"| Optimal threshold | {emp_result['optimal_threshold']:.2f} |",
        f"| Churn rate in test set | {emp_result['churn_rate']*100:.1f}% |",
        f"| Customers | {emp_result['n_customers']:,} |",
        f"| Churners | {emp_result['n_churners']:,} |",
        "",
        "### Profit Comparison",
        f"| Strategy | Net Profit |",
        f"|---|---:|",
        f"| **Model-guided campaign (EMP)** | **${emp_result['emp']:,.2f}** |",
        f"| Contact everyone | ${emp_result['baseline_contact_all']:,.2f} |",
        f"| Contact nobody | ${emp_result['baseline_contact_none']:,.2f} |",
        f"| Random 50% campaign | ${emp_result['random_campaign_profit']:,.2f} |",
        f"| **Lift over random** | **{emp_result['lift_over_random']:.2f}×** |",
        "",
        "### Retention Programme ROI",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Customers contacted | {roi_result['customers_contacted']:,} |",
        f"| Churners saved (TP) | {roi_result['churners_saved']:,} |",
        f"| Churners missed (FN) | {roi_result['churners_missed']:,} |",
        f"| Total spend | ${roi_result['total_spend']:,.2f} |",
        f"| Net profit | ${roi_result['net_profit']:,.2f} |",
        f"| **ROI** | **{roi_result['roi_pct']:.1f}%** |",
    ]
    return "\n".join(lines)
