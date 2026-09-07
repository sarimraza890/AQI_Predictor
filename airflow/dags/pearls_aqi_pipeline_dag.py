from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "pearls_mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "pearls_aqi_hourly_feature_pipeline",
    default_args=default_args,
    description="Ingest weather/pollutants and engineer features hourly for Karachi, Lahore, Islamabad",
    schedule_interval="0 * * * *",
    catchup=False,
    tags=["aqi", "feature_store", "serverless"],
) as hourly_dag:

    cities = ["Karachi", "Lahore", "Islamabad"]
    feature_tasks = []

    for city in cities:
        task = BashOperator(
            task_id=f"ingest_features_{city.lower()}",
            bash_command=f"python -m feature_pipeline.run --city {city}",
        )
        feature_tasks.append(task)

with DAG(
    "pearls_aqi_daily_training_pipeline",
    default_args=default_args,
    description="Retrain Ridge, RF, GBM, and MLP models daily and promote best to production",
    schedule_interval="30 2 * * *",
    catchup=False,
    tags=["aqi", "training", "model_registry"],
) as daily_dag:

    train_tasks = []
    for city in cities:
        train_task = BashOperator(
            task_id=f"train_models_{city.lower()}",
            bash_command=f"python -m training_pipeline.train --city {city}",
        )
        train_tasks.append(train_task)
