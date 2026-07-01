# SceneForge Deployment Guide

## Phase 1 — Local (Docker Compose)

```bash
# Copy env template
cp .env.example .env
# Add your OPENAI_API_KEY

docker compose -f docker/docker-compose.yml up --build
```

| Service    | URL                        |
|------------|----------------------------|
| API        | http://localhost:8000       |
| Swagger    | http://localhost:8000/docs  |
| Streamlit  | http://localhost:8501       |
| Prometheus | http://localhost:9090       |
| Grafana    | http://localhost:3000       |
| MLflow     | http://localhost:5001       |

---

## Phase 2 — AWS EC2

```bash
# 1. Launch g4dn.xlarge instance (Deep Learning AMI)
# 2. SSH in and clone the repo
git clone https://github.com/<you>/sceneforge.git && cd sceneforge

# 3. Build and push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com

docker build -f docker/Dockerfile -t sceneforge-api .
docker tag sceneforge-api:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/sceneforge-api:latest
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/sceneforge-api:latest

# 4. Run on EC2
docker compose -f docker/docker-compose.yml up -d
```

---

## Phase 3 — AWS EKS

```bash
# 1. Create cluster
eksctl create cluster \
  --name sceneforge \
  --region us-east-1 \
  --nodegroup-name gpu-nodes \
  --node-type g4dn.xlarge \
  --nodes 2 --nodes-min 1 --nodes-max 4

# 2. Create secrets
kubectl create secret generic sceneforge-secrets \
  --from-literal=db-password=<PW> \
  --from-literal=openai-api-key=<KEY> \
  --from-literal=grafana-password=admin \
  -n sceneforge

# 3. Deploy all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/data-services.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/monitoring.yaml

# 4. Verify
kubectl get pods -n sceneforge
kubectl get svc  -n sceneforge
```