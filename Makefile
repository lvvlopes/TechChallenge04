# Makefile — atalhos de desenvolvimento
# (No Windows, use `make` via Git Bash/WSL, ou rode os comandos manualmente.)

.PHONY: help install install-dev data demo api test lint format typecheck clean

help:
	@echo "Alvos disponíveis:"
	@echo "  install       Instala dependências de runtime"
	@echo "  install-dev   Instala dependências de desenvolvimento + pacote editável"
	@echo "  data          Gera datasets sintéticos em data/samples/"
	@echo "  demo          Executa a demonstração multimodal"
	@echo "  api           Sobe a API REST (uvicorn, reload)"
	@echo "  test          Roda a suíte de testes (pytest)"
	@echo "  lint          Verifica estilo (ruff)"
	@echo "  format        Formata o código (ruff)"
	@echo "  typecheck     Checa tipos (mypy)"
	@echo "  clean         Remove caches e artefatos"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

data:
	python scripts/generate_synthetic_data.py

demo:
	python scripts/run_demo.py

api:
	uvicorn multimodal_monitor.api.main:app --reload --app-dir src

test:
	pytest

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage outputs
	find . -type d -name __pycache__ -exec rm -rf {} +
