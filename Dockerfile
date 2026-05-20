FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir --upgrade -r requirements-api.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.saas_app:app --host 0.0.0.0 --port ${PORT:-7860}"]
