# Use official slim Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything from your project into the container
COPY . /app

# Install system build dependencies if needed
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set Python path inside container (optional but good practice)
ENV PYTHONPATH=/app

# Default command: run your pipeline
CMD ["python", "-m", "src.main"]
