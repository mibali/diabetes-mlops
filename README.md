
# 🩺 Diabetes Prediction API — MLOps Project

A complete **MLOps demonstration project** featuring model training, experiment tracking, model registry, FastAPI serving, Docker containerization, automated testing, and CI/CD automation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Docker](https://img.shields.io/badge/Docker-✅-blue)
![MLflow](https://img.shields.io/badge/MLflow-Model_Registry-orange)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-success)

---

## ✨ Features

- MLflow **experiment tracking**
- MLflow **Model Registry** integration
- FastAPI REST API with Pydantic validation
- Dockerized deployment
- GitHub Actions CI/CD pipeline
- Automated testing with `pytest`
- Reproducible model training and serving workflow

---

## 🚀 Quick Start

### 1. Local Setup

```bash
git clone <your-repo-url>
cd diabetes-mlops

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Start the API locally:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The API should now be running at:

```text
http://127.0.0.1:8000
```

---

## 🐳 Docker

Build the Docker image:

```bash
docker build -t diabetes-api .
```

Run the container:

```bash
docker run -p 8000:8000 diabetes-api
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

## 🧪 Test Prediction

Send a sample prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 62,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.127,
    "Age": 47
  }'
```

Example response:

```json
{
  "prediction": 1,
  "result": "Diabetic"
}
```

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | scikit-learn, pandas, numpy |
| Experiment Tracking | MLflow Tracking |
| Model Registry | MLflow Model Registry |
| API Framework | FastAPI + Pydantic |
| Serving | Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## 📁 Project Structure

```bash
diabetes-mlops/
├── src/
│   ├── train.py              # Training script with MLflow logging
│   └── inference.py          # Model loading and prediction logic
├── api/
│   └── main.py               # FastAPI application
├── models/
│   └── diabetes_model.pkl    # Serialized trained model
├── tests/
│   └── test_api.py           # API tests
├── .github/
│   └── workflows/
│       └── ci-cd.yml         # GitHub Actions pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 MLflow

This project uses MLflow for experiment tracking and model registry.

| Item | Value |
|---|---|
| Experiment | `diabetes-prediction` |
| Registered Model | `diabetes-random-forest` |

Start the MLflow UI locally:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

## ✅ What This Project Demonstrates

This project demonstrates a practical end-to-end MLOps workflow, including:

- Reproducible model training
- Experiment tracking with MLflow
- Model versioning and registry
- Production-ready API serving
- Automated testing
- CI/CD automation
- Docker-based deployment
- Clear separation between training, inference, API, and testing layers

---

## 🔮 Future Enhancements

Potential improvements include:

- Data versioning with DVC
- Model monitoring and drift detection
- Cloud deployment on AWS, Render, or Heroku
- Pipeline orchestration with Prefect or Airflow
- Feature store integration
- Automated model promotion between registry stages
- Shadow deployment or canary release support

---

## 🧑‍💻 Author

Built as an MLOps prod project for deploying a machine learning model as a production-ready API.
````
