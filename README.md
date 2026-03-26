# Customer Churn Prediction MLOps Project 🚀

This project demonstrates a production-grade MLOps pipeline for predicting customer churn using **XGBoost/LightGBM**, containerized with **Docker**, and deployed on **AWS (ECS, RDS, and CodePipeline)**.

## 📂 Project Structure
Following the professional `src` layout:
- `src/churn_app/`: Core Flask API and database logic.
- `models/`: Serialized model artifacts (.pkl).
- `terraform/`: Infrastructure as Code (IaC) for AWS provisioning.
- `buildspec.yml`: Instructions for AWS CodeBuild.
- `requirements.txt`: Python dependencies.

## 🏗️ Architecture
1. **CI/CD**: GitHub → AWS CodePipeline → AWS CodeBuild.
2. **Containerization**: Docker images stored in Amazon ECR.
3. **Serving**: Flask API running on AWS ECS (Fargate).
4. **Data**: Amazon RDS (PostgreSQL) for customer records.

## 🚀 How to Run Locally
1. Clone the repo: `git clone https://github.com/Armaghan6623/Customer-Churn.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python src/churn_app/app.py`

---
*Developed by Armaghan - MLOps Engineer in Training*
