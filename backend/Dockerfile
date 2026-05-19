FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and create cache directory in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better cache)
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ai module and backend code
COPY ai ./ai
COPY backend ./backend

# Expose port
EXPOSE 8080

# Run the FastAPI backend
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
