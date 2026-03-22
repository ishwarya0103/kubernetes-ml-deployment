# Kubernetes-Based ML Model Deployment (CIFAR-10)

An end-to-end ML system that trains image classification models, tracks experiments, deploys an API, and scales it using Kubernetes with monitoring and orchestration.


## Overview

Pipeline:

Data → Training → MLflow → FastAPI → Docker → Kubernetes → HPA → Prometheus/Grafana → Airflow

Models:
- CNN (PyTorch)
- Random Forest (scikit-learn)


## Tech Stack

- PyTorch, scikit-learn  
- FastAPI  
- MLflow  
- Docker  
- Kubernetes (HPA)  
- Prometheus + Grafana  
- Airflow  


## Setup

```bash
git clone <repo-url>
cd kubernetes-ml-deployment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training + Tracking

```bash
mlflow server --port 5000

python src/train_cnn.py
python src/train_rf.py
```
Open: http://127.0.0.1:5000


## Run API

```bash
uvicorn api.app:app --reload
```
Docs: http://127.0.0.1:8000/docs


## Docker

```bash
docker build -t cifar10-api .
docker run -p 8000:8000 cifar10-api
```


## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

Check:
```bash
kubectl get pods
kubectl get hpa
```


## Monitoring

```bash
helm install kube-prom-stack prometheus-community/kube-prometheus-stack   -n monitoring --create-namespace   --set nodeExporter.enabled=false

kubectl port-forward svc/kube-prom-stack-grafana 3000:80 -n monitoring
```

Grafana: http://localhost:3000

Example query:
```
sum(rate(http_request_duration_seconds_count[1m]))
```

## Airflow Pipeline

```bash
airflow standalone
```

UI: http://localhost:8080

Pipeline:
```
train_cnn → train_rf → compare_results
```


## Key Highlights

- End-to-end ML system (training → deployment → monitoring → orchestration)
- Kubernetes autoscaling with HPA
- Real-time metrics using Prometheus + Grafana
- Experiment tracking with MLflow
- Pipeline automation with Airflow


