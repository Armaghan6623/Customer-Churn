---
title: Customer Churn Prediction API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# customer_churn_saas

Dockerized Streamlit app for customer churn risk profiling and macro forecasting.

## Local run (without Docker)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Local run (Docker)

```bash
docker build -t customer-crunch .
docker run -p 7860:7860 customer-crunch
```

Open http://localhost:7860

## Layout

- `app.py` — Streamlit entry point
- `src/customer_churn_saas/` — application package
- `classification/`, `forecasting/` — ML modules (bundled at deploy time)
- `saved_models/` — serialized model artifacts
