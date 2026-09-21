# AI Multi-Agent Student Super Assistant

This is a production-ready AI Student Assistant powered by LangGraph, FastAPI, and Gradio.

## Architecture

- **Frontend**: Gradio for responsive and interactive UI (`http://localhost:7860`).
- **Backend API**: FastAPI to expose LLM and orchestration endpoints (`http://localhost:8000`).
- **Agent Orchestrator**: LangGraph to route and handle multi-agent tasks (Study, Coding, Exam, Project, Document, Audio, 3D Video).
- **Audio/Video Engines**: Custom modules using `edge-tts` and `moviepy` to generate audio and 3D explainer videos from text.
- **Database**: MySQL for structured data and ChromaDB/FAISS (via LangChain) for RAG document storage.
- **Infrastructure**: Fully Dockerized and orchestrated using Kubernetes.

## Deployment Guide

### Local Development
1. Run local frontend directly:
   ```bash
   cd frontend
   python app.py
   ```
   Access Gradio UI at `http://localhost:7860`.

### Local Development (Docker Compose)
1. Copy `.env.example` to `.env` and fill in your configuration.
2. Run `docker-compose -f docker/docker-compose.yml up --build`
3. Access Frontend at `http://localhost:7860`
4. Access Backend API at `http://localhost:8000/docs`

### Production Deployment (Kubernetes)
Ensure you have `kubectl` configured with your cluster.

1. Create Namespace:
   `kubectl apply -f k8s/namespace.yaml`
2. Update `k8s/secrets.yaml` with base64 encoded API keys.
3. Deploy dependencies:
   `kubectl apply -f k8s/pvc.yaml`
   `kubectl apply -f k8s/secrets.yaml`
   `kubectl apply -f k8s/configmap.yaml`
   `kubectl apply -f k8s/mysql-deployment.yaml`
   `kubectl apply -f k8s/redis-deployment.yaml`
4. Deploy the application:
   `kubectl apply -f k8s/backend-deployment.yaml`
   `kubectl apply -f k8s/worker-deployment.yaml`
   `kubectl apply -f k8s/frontend-deployment.yaml`
5. Expose services:
   `kubectl apply -f k8s/services.yaml`
   `kubectl apply -f k8s/ingress.yaml`
   `kubectl apply -f k8s/hpa.yaml`

To check the status:
`kubectl get pods -n student-assistant`
"# Student_AI_Study_Multi_Assistant_Testing1" 
