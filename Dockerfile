FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    pydantic==2.6.4 \
    openai==1.30.0

# Copy all environment files
COPY env.py .
COPY app.py .
COPY openenv.yaml .
COPY inference.py .

# Expose port required by HuggingFace Spaces
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
