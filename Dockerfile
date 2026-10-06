# House Price Predictor — Streamlit demo
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY data ./data
COPY artifacts ./artifacts

RUN pip install --no-cache-dir -e ".[demo]" \
    && pip install --no-cache-dir lightgbm joblib

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
