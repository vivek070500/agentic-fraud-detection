FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Generate data + train model at build time
RUN python -m data.generate_dataset && python -m ml.train_model

# Expose ports: 8000 (API), 8501 (Streamlit)
EXPOSE 8000 8501

# Startup script runs both services
COPY deploy/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
