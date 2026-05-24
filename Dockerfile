FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AGENT_ENABLE_LLM=false \
    AGENT_DEFAULT_PROVIDER=mock \
    LLM_PROVIDER_ORDER=mock,gemini,grok \
    LLM_ENABLE_FALLBACK=true

COPY requirements.txt requirements-dev.txt pyproject.toml README.md .env.example ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY configs/ ./configs/

COPY runtime_assets/artifacts/ ./artifacts/
COPY runtime_assets/configs/ ./configs/
COPY runtime_assets/data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
