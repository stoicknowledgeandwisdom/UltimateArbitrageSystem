FROM apache/airflow:2.7.0-python3.9

# Switch to root to install system dependencies
USER root

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements and install Python dependencies
COPY requirements.txt /opt/airflow/
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Copy DAGs, plugins, and configuration
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root plugins/ /opt/airflow/plugins/
COPY --chown=airflow:root config/ /opt/airflow/config/
COPY --chown=airflow:root sensors/ /opt/airflow/dags/sensors/
COPY --chown=airflow:root operators/ /opt/airflow/dags/operators/
COPY --chown=airflow:root hooks/ /opt/airflow/dags/hooks/
COPY --chown=airflow:root utils/ /opt/airflow/dags/utils/

# Set environment variables
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__PLUGINS_FOLDER=/opt/airflow/plugins
ENV AIRFLOW__CORE__BASE_LOG_FOLDER=/opt/airflow/logs
ENV AIRFLOW__CORE__EXECUTOR=KubernetesExecutor
ENV AIRFLOW__KUBERNETES__NAMESPACE=airflow
ENV AIRFLOW__KUBERNETES__WORKER_CONTAINER_REPOSITORY=arbitrage-airflow
ENV AIRFLOW__KUBERNETES__WORKER_CONTAINER_TAG=latest

# Create necessary directories
RUN mkdir -p /opt/airflow/logs /opt/airflow/dags /opt/airflow/plugins

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=5 \
    CMD airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"

