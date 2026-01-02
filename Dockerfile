FROM python:3.13-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app AFTER deps
COPY . .

EXPOSE 8000

# ✅ PURE EXEC FORM (NO SHELL = NO SYNTAX ERRORS)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]