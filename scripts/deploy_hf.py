"""Deploy the Gradio Docker Space bundle to Hugging Face."""
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / ".hf-deploy-staging"
DEFAULT_SPACE = "MuhammadArmaghan/customer-churn"


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def stage_bundle() -> Path:
    crunch = REPO_ROOT / "customer crunch"
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir()

    for name in ("Dockerfile", "README.md", ".dockerignore"):
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy2(src, STAGING / name)

    dst_crunch = STAGING / "customer crunch"
    shutil.copytree(crunch / "ui", dst_crunch / "ui")
    shutil.copytree(crunch / "agent", dst_crunch / "agent")
    shutil.copytree(crunch / "aiops", dst_crunch / "aiops")
    shutil.copytree(crunch / "classification", dst_crunch / "classification")

    data_src = crunch / "data" / "raw" / "Churn_Modelling kaggel.csv"
    if data_src.exists():
        data_dst = dst_crunch / "data" / "raw"
        data_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_src, data_dst / data_src.name)

    shutil.copy2(crunch / "requirements-hf.txt", dst_crunch / "requirements-hf.txt")

    dst_models = dst_crunch / "saved_models"
    dst_models.mkdir()
    model = crunch / "saved_models" / "churn_pipeline.joblib"
    if not model.exists():
        raise FileNotFoundError(f"Missing model artifact: {model}")
    shutil.copy2(model, dst_models / "churn_pipeline.joblib")

    for pycache in STAGING.rglob("__pycache__"):
        shutil.rmtree(pycache)

    return STAGING


def deploy(token: str, space_id: str) -> None:
    staging = stage_bundle()
    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name', who)}")

    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    upload_folder(
        folder_path=str(staging),
        repo_id=space_id,
        repo_type="space",
        token=token,
        commit_message="Deploy Customer Crunch agents (Advisor + MLOps)",
        delete_patterns=["*"],
    )
    slug = space_id.replace(" ", "%20")
    print(f"Deployed to https://huggingface.co/spaces/{slug}")
    print("Space is building — check the Logs tab if the app does not load within a few minutes.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--space", default=os.environ.get("HF_SPACE_ID", DEFAULT_SPACE))
    args = parser.parse_args()
    if not args.token:
        print("Set HF_TOKEN or pass --token", file=sys.stderr)
        sys.exit(1)
    deploy(args.token, args.space)


if __name__ == "__main__":
    main()
