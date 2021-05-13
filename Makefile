build:
	docker compose build

start:
	docker compose up

mlflow-api/connect:
	docker compose exec mlflow_api /bin/bash

run/experiment:
	docker compose exec mlflow_api python src/main.py

uninstall:
	docker compose down