"""Deploy the staged Space bundle to Hugging Face."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_SRC = REPO_ROOT / "customer crunch" / "customer_churn_saas"
STAGING = REPO_ROOT / ".hf-deploy-staging"
SPACE_REPO = "MuhammadArmaghan/customer-crunch"


def stage_bundle() -> Path:
    crunch = REPO_ROOT / "customer crunch"
    if STAGING.exists():
        shutil.rmtree(STAGING)
    shutil.copytree(APP_SRC, STAGING)

    shutil.copytree(crunch / "classification", STAGING / "classification")
    shutil.copytree(crunch / "forecasting", STAGING / "forecasting")

    src_models = crunch / "saved_models"
    dst_models = STAGING / "saved_models"
    dst_models.mkdir(exist_ok=True)
    for name in (
        "churn_pipeline.joblib",
        "forecast_model.joblib",
        "xgb_forecast_model.joblib",
        "forecast_plot.png",
    ):
        src = src_models / name
        if src.exists():
            shutil.copy2(src, dst_models / name)

    trends = crunch / "monthly_churn_trends.csv"
    if not trends.exists():
        trends = crunch / "data" / "raw" / "monthly_churn_trends.csv"
    if trends.exists():
        shutil.copy2(trends, STAGING / "monthly_churn_trends.csv")

    for pycache in STAGING.rglob("__pycache__"):
        shutil.rmtree(pycache)
    for pyc in STAGING.rglob("*.pyc"):
        pyc.unlink()

    return STAGING


def deploy(token: str) -> None:
    staging = stage_bundle()
    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token)
    api.create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    upload_folder(
        folder_path=str(staging),
        repo_id=SPACE_REPO,
        repo_type="space",
        token=token,
        commit_message="Deploy Docker Streamlit app from local script",
        delete_patterns=["*"],
    )
    print(f"Deployed to https://huggingface.co/spaces/{SPACE_REPO}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    if not args.token:
        print("Set HF_TOKEN or pass --token", file=sys.stderr)
        sys.exit(1)
    deploy(args.token)


if __name__ == "__main__":
    main()
