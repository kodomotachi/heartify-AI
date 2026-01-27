

.PHONY: help build up up-gpu down logs shell test clean restart

# Default target
help:
	@echo "FastAPI Docker Commands:"
	@echo ""
	@echo "  make build       - Build Docker image"
	@echo "  make up          - Start services (CPU mode)"
	@echo "  make up-gpu      - Start services (GPU mode)"
	@echo "  make up-storage  - Start with MinIO storage"
	@echo "  make down        - Stop services"
	@echo "  make logs        - View logs"
	@echo "  make shell       - Open shell in container"
	@echo "  make test        - Run API tests"
	@echo "  make restart     - Restart services"
	@echo "  make clean       - Remove containers and volumes"
	@echo ""

# Build image
build:
	@echo "Building Docker image..."
	docker compose build

# Start services (CPU)
up:
	@echo "Starting services (CPU mode)..."
	docker compose up -d
	@echo "Services started!"
	@echo "API: http://localhost:8080"
	@echo "Docs: http://localhost:8080/docs"

# Start services (GPU)
up-gpu:
	@echo "Starting services (GPU mode)..."
	@DEVICE=gpu:0 docker compose up -d
	@echo "Services started with GPU!"
	@echo "API: http://localhost:8080"

# Start with storage
up-storage:
	@echo "Starting services with MinIO..."
	docker compose --profile storage up -d
	@echo "Services started!"
	@echo "API: http://localhost:8080"
	@echo "MinIO: http://localhost:9001"

# Stop services
down:
	@echo "Stopping services..."
	docker compose down

# View logs
logs:
	docker compose logs -f fastapi

# Open shell
shell:
	docker compose exec fastapi /bin/bash

# Run tests
test:
	@echo "Running API tests..."
	docker compose exec fastapi python app/api/test_api.py

# Check health
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8080/health | python -m json.tool || echo "Service not responding"

# Restart services
restart:
	@echo "Restarting services..."
	docker compose restart fastapi

# Clean up
clean:
	@echo "Cleaning up..."
	docker compose down -v
	docker system prune -f
	@echo "Cleanup complete!"

# Full rebuild
rebuild:
	@echo "Rebuilding from scratch..."
	docker compose down
	docker compose build --no-cache
	docker compose up -d
	@echo "Rebuild complete!"

# Quick start (build + run)
start: build up

# Development mode (with live reload)
dev:
	docker compose up

# View resource usage
stats:
	docker stats heartify-fastapi

# GPU check
gpu-check:
	@echo " Checking GPU availability..."
	docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi || echo " GPU not available"