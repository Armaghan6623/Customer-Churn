# HF Deployment & CI/CD Setup Guide

## ✅ Completed Steps

1. **Code Corrections**
   - ✅ Fixed filename: `dataclening.py` → `datacleaning.py`
   - ✅ Added comprehensive docstrings to all functions
   - ✅ Created proper preprocessing package structure
   - ✅ Improved code comments

2. **Git Commit**
   - ✅ Staged and committed all changes
   - ✅ Pushed to main branch

3. **CI/CD Workflows Ready**
   - ✅ `ci-flake8.yml` - Code style linting
   - ✅ `hf_deploy.yml` - Auto-deploy to HF on push to main
   - ✅ `mlops-drift.yml` - Drift monitoring pipeline

## 🚀 Required GitHub Secrets & Variables

To activate CI/CD deployment, configure these in your GitHub repository:

### GitHub Secrets (Settings → Secrets and variables → Actions)
```
HF_SPACE_DEPLOY_TOKEN: <your-huggingface-api-token>
```

Obtain your HF token from: https://huggingface.co/settings/tokens

### GitHub Variables (Settings → Secrets and variables → Variables)
```
HF_SPACE_ID: MuhammadArmaghan/customer-churn
```

## 📦 Manual Deployment (if needed)

If you need to manually deploy without waiting for CI/CD:

```bash
pip install huggingface_hub
export HF_TOKEN=<your-token>
python scripts/deploy_hf.py --token $HF_TOKEN --space MuhammadArmaghan/customer-churn
```

## 🔄 How the CI/CD Pipeline Works

1. **Push to main** → Triggers `hf_deploy.yml`
2. **Workflow runs** → Stages Docker bundle with all models and code
3. **Uploads to HF** → Creates/updates Space with new code and models
4. **Space rebuilds** → Docker container starts on Hugging Face
5. **App goes live** → Gradio UI becomes available at https://huggingface.co/spaces/MuhammadArmaghan/customer-churn

## 🐳 Docker Deployment Details

- **Base image**: Python 3.10-slim
- **App port**: 7860 (Gradio)
- **Models included**: 
  - `churn_pipeline.joblib` (main classifier)
  - `forecast_model.joblib` (forecasting)
  - `xgb_forecast_model.joblib` (XGBoost forecasting)
- **Additional features**:
  - SHAP explainability visualizations
  - MLOps agent for monitoring
  - Drift detection using KS test

## 📝 Next Steps

1. Go to GitHub repository settings
2. Add `HF_SPACE_DEPLOY_TOKEN` secret with your HF API token
3. Verify `HF_SPACE_ID` variable is set
4. Make any code changes and push to `main` branch
5. Deployment automatically triggers! ✨

## 🔧 Dockerfile Configuration

The `Dockerfile` in the repo defines:
- Working directory: `/app`
- Installs: `requirements-hf.txt` dependencies
- Exposes: Port 7860 for Gradio UI
- Entrypoint: Runs `customer_crunch/ui/app.py`

## ✨ Project Structure (Deployed)

```
/app/
├── customer_crunch/
│   ├── ui/
│   │   └── app.py              # Gradio interface
│   ├── classification/         # Model training/prediction
│   ├── agent/                  # MLOps monitoring agent
│   ├── aiops/                  # Drift detection
│   ├── forecasting/            # Time series forecasting
│   ├── preprocessing/          # Data cleaning utilities
│   ├── saved_models/           # Trained model artifacts
│   └── data/                   # Reference datasets
├── Dockerfile
├── README.md
└── .dockerignore
```

---

**Status**: ✅ Ready for automatic CI/CD deployment on next push!
