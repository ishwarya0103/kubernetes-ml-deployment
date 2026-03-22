from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

PROJECT_DIR = "/Users/ishwarya/Desktop/kubernetes-ml-deployment"

COMMON_ENV = (
    'export no_proxy="*"; '
    'export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES; '
    'export PYTHONFAULTHANDLER=1; '
)

with DAG(
    dag_id="cifar10_ml_pipeline",
    start_date=datetime(2026, 3, 22),
    schedule=None,
    catchup=False,
    tags=["ml", "cifar10"],
) as dag:

    train_cnn = BashOperator(
        task_id="train_cnn",
        bash_command=(
            f'{COMMON_ENV}'
            f'cd "{PROJECT_DIR}" && '
            f'"{PROJECT_DIR}/venv/bin/python" src/train_cnn.py'
        ),
    )

    train_rf = BashOperator(
        task_id="train_rf",
        bash_command=(
            f'{COMMON_ENV}'
            f'cd "{PROJECT_DIR}" && '
            f'"{PROJECT_DIR}/venv/bin/python" src/train_rf.py'
        ),
    )

    compare_results = BashOperator(
        task_id="compare_results",
        bash_command=(
            f'{COMMON_ENV}'
            f'cd "{PROJECT_DIR}" && '
            f'"{PROJECT_DIR}/venv/bin/python" src/compare_results.py'
        ),
    )

    train_cnn >> train_rf >> compare_results