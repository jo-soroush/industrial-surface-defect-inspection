FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt pyproject.toml README.md .env.example ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
