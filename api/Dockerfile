# ---------- 1️⃣  build stage: install Python deps -----------------
    FROM --platform=$BUILDPLATFORM python:3.11-slim-bookworm AS build

    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential git && \
        rm -rf /var/lib/apt/lists/*
    
    WORKDIR /code
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---------- 2️⃣  runtime stage: slim final image -----------------
    FROM python:3.11-slim-bookworm
    
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH=/app            
    
    WORKDIR /app
    
    # ── copy site-packages and binaries from build layer
    COPY --from=build /usr/local/lib/python3.11/site-packages \
                      /usr/local/lib/python3.11/site-packages
    COPY --from=build /usr/local/bin /usr/local/bin
    
    # ── copy your API code
    COPY . /app                    
    
  
    
    EXPOSE 8000
    HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
    
    CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
         "main:app", "--bind", "0.0.0.0:8000"]
    
