.PHONY: help install install-dev test test-all test-coverage lint format clean docker-build docker-run docker-stop docker-logs serve

help:
	@echo "Sentiment Analysis V2 - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install base dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run pytest"
	@echo "  make test-all         - Run all tests with coverage"
	@echo "  make test-coverage    - Generate HTML coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run flake8 linter"
	@echo "  make format           - Format code with black"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-stop      - Stop Docker container"
	@echo "  make docker-logs      - View Docker logs"
	@echo "  make docker-test      - Run tests in Docker"
	@echo ""
	@echo "Server:"
	@echo "  make serve            - Start Flask development server"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove generated files"
	@echo ""

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest v2/tests/ -v

test-all:
	pytest v2/tests/ -v --tb=short --cov=v2 --cov-report=term-missing

test-coverage:
	pytest v2/tests/ --cov=v2 --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 v2/ --max-line-length=120 --exclude=__pycache__

format:
	black v2/
	isort v2/

docker-build:
	docker build -t sentiment-api:v2 .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f sentiment-api

docker-test:
	docker build -t sentiment-api:test . && \
	docker run --rm --entrypoint pytest sentiment-api:test v2/tests/ -v

serve:
	python -m v2.cli serve --host 127.0.0.1 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf build dist *.egg-info
	rm -rf logs/*.log
	rm -rf .mypy_cache

all: install test docker-build

.DEFAULT_GOAL := help
