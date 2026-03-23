FROM prefecthq/prefect:3-latest
WORKDIR /app
COPY flows/ ./flows/
COPY requirements.txt .
COPY prefect.yaml .
RUN pip install -r requirements.txt
ENV PYTHONPATH=/app
