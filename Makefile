.PHONY: setup run quick test app mlflow clean

setup:      ## create env and install pinned deps
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pip install -e .

run:        ## full pipeline (about 1-2 min on CPU)
	python -m predmaint.pipeline

quick:      ## smoke run with 5 Optuna trials
	python -m predmaint.pipeline --trials 5

test:
	pytest -q

app:        ## launch the Gradio demo on http://127.0.0.1:7860
	python app/app.py

mlflow:     ## browse experiments and the model registry
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf mlflow.db mlruns .pytest_cache
