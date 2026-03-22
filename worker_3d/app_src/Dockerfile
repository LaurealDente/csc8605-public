FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 and tifffile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY conf/ /app/conf/

CMD ["python", "-m", "app.server"]
