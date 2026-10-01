FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]