FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for organized development
RUN mkdir -p /app/src /app/data /app/notebooks

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command
CMD ["python"]