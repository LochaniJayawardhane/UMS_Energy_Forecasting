FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs data

# Create entrypoint script
RUN echo '#!/bin/bash\n\
case "$1" in\n\
  "api")\n\
    echo "Starting API server..."\n\
    uvicorn src.main:app --host 0.0.0.0 --port 8000\n\
    ;;\n\
  "worker")\n\
    echo "Starting Dramatiq worker..."\n\
    python -m src.dramatiq_worker\n\
    ;;\n\
  *)\n\
    exec "$@"\n\
    ;;\n\
esac' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["api"] 