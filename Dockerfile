# 1. Base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (for scikit-learn, numpy etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first for caching
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy entire project
COPY . .

# 7. Expose port
EXPOSE 8000

# 8. Run FastAPI (uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
