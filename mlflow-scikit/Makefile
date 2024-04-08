install:
	@echo "Installing packed..."
	
	poetry env use $(shell which python3.10) && \
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install

env:
	poetry shell

mlflow_ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

run:
	poetry run python main.py