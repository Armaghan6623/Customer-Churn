"""Deploy the Gradio Docker Space bundle to Hugging Face."""
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / ".hf-deploy-staging"
DEFAULT_SPACE = "MuhammadArmaghan/customer-churn"


def stage_bundle() -> Path:
    crunch = REPO_ROOT / "customer_crunch"
    if not crunch.is_dir():
        raise FileNotFoundError(f"customer_crunch package not found at: {crunch}")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()

    # Top-level deployment files
    for name in ("Dockerfile", "README.md", ".dockerignore"):
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy2(src, STAGING / name)

    # Core package modules
    dst_crunch = STAGING / "customer_crunch"
    for module in ("ui", "agent", "aiops", "classification", "forecasting"):
        src_mod = crunch / module
        if src_mod.is_dir():
            shutil.copytree(src_mod, dst_crunch / module)

    # Package init
    init = crunch / "__init__.py"
    if init.exists():
        dst_crunch.mkdir(exist_ok=True)
        shutil.copy2(init, dst_crunch / "__init__.py")

    # requirements
    req_src = crunch / "requirements-hf.txt"
    if req_src.exists():
        shutil.copy2(req_src, dst_crunch / "requirements-hf.txt")

    # Saved models (required for inference)
    src_models = crunch / "saved_models"
    dst_models = dst_crunch / "saved_models"
    dst_models.mkdir(parents=True, exist_ok=True)

    required_models = ["churn_pipeline.joblib", "forecast_model.joblib", "xgb_forecast_model.joblib"]
    for model_file in required_models:
        model_path = src_models / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Missing required model artifact: {model_path}")
        shutil.copy2(model_path, dst_models / model_file)

    # Optional artefacts (plots, retrain checkpoint)
    for optional in ["forecast_plot.png", "retrained_chk_test.joblib",
                     "shap_bar.png", "shap_force.png", "shap_waterfall.png"]:
        src = src_models / optional
        if src.exists():
            shutil.copy2(src, dst_models / optional)

    # Training/reference data (needed for drift monitoring)
    data_src = crunch / "data"
    if data_src.is_dir():
        shutil.copytree(data_src, dst_crunch / "data")

    # Strip __pycache__ to keep the upload lean
    for pycache in STAGING.rglob("__pycache__"):
        shutil.rmtree(pycache)

    print(f"Staged bundle at: {STAGING}")
    return STAGING


def deploy(token: str, space_id: str) -> None:
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    staging = stage_bundle()

    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name', who)}")

    print(f"Creating/verifying space: {space_id}")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    print("Uploading files to Hugging Face Space...")
    upload_folder(
        folder_path=str(staging),
        repo_id=space_id,
        repo_type="space",
        token=token,
        commit_message="deploy: Customer Crunch — SHAP explainability, advisor agent, MLOps agent",
        delete_patterns=["*"],
    )

    slug = space_id.replace(" ", "%20")
    print(f"\nDeployed to: https://huggingface.co/spaces/{slug}")
    print("Space is building — check the Logs tab if the app does not load within a few minutes.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Customer Crunch to Hugging Face Spaces")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--space", default=os.environ.get("HF_SPACE_ID", DEFAULT_SPACE),
                        help="Space ID e.g. username/space-name")
    args = parser.parse_args()

    if not args.token:
        print("Error: provide --token or set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    deploy(args.token, args.space)


if __name__ == "__main__":
    main()
