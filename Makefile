.PHONY: serve-mlflow serve-triton

serve-mlflow:
	poetry run mlflow models serve -m models/meta_model -h 0.0.0.0 -p 5001

serve-triton:
	docker-compose -f docker-compose.triton.yml up
