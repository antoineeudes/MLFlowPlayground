build:
	docker-compose build

start:
	docker-compose up

run-experiment:
	docker-compose exec -it mlflow python src/main.python

uninstall:
	docker-compose down