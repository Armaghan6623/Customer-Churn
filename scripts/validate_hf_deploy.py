"""Local validation for the Hugging Face Gradio deployment."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "customer crunch" / "saved_models" / "churn_pipeline.joblib"
APP = REPO_ROOT / "customer crunch" / "ui" / "app.py"


def check_model() -> None:
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL}")
    import joblib

    joblib.load(MODEL)
    print(f"[OK] Model loads: {MODEL}")


def check_app_import() -> None:
    try:
        import gradio  # noqa: F401
    except ImportError:
        print("[SKIP] gradio not installed locally - run: pip install -r \"customer crunch/requirements.txt\"")
        return

    sys.path.insert(0, str(REPO_ROOT / "customer crunch" / "ui"))
    os.environ.setdefault("CHURN_PIPELINE_PATH", str(MODEL))
    import importlib.util

    spec = importlib.util.spec_from_file_location("churn_app", APP)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import app module: {APP}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.PIPELINE is None:
        raise RuntimeError(f"Pipeline failed to load: {module.PIPELINE_LOAD_ERROR}")
    print("[OK] Gradio app imports and pipeline is loaded")


def check_docker() -> None:
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print("[SKIP] Docker not available - start Docker Desktop, then run:")
        print("       docker build -t customer-churn:latest .")
        print("       docker run --rm -p 7860:7860 customer-churn:latest")
        return

    print("[OK] Docker daemon is running")
    subprocess.run(
        ["docker", "build", "-t", "customer-churn:latest", str(REPO_ROOT)],
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "customer-churn:latest",
            "python",
            "-c",
            (
                "import joblib, os; "
                "p='customer_crunch/saved_models/churn_pipeline.joblib'; "
                "assert os.path.exists(p), p; joblib.load(p); "
                "print('Model loaded OK inside container')"
            ),
        ],
        check=True,
    )
    print("[OK] Docker image builds and model loads in container")


def main() -> None:
    check_model()
    check_app_import()
    check_docker()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
