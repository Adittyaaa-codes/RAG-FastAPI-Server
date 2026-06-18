# Use the official Python lightweight image
FROM python:3.13-slim

# Set environment variables to optimize Python execution
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Tell fastembed to save models inside the container image so they don't re-download on restart
ENV FASTEMBED_CACHE_DIR=/app/model_cache

# Set the working directory
WORKDIR /app

# Install uv (astral-sh) for extremely fast dependency resolution and installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the dependency management files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv into the system environment
RUN uv sync --frozen --no-dev

# Copy the rest of your application code
COPY . /app

# ---------------------------------------------------------
# PRE-DOWNLOAD THE EMBEDDING MODELS
# This is a critical best practice for cloud deployments!
# By running this during the Docker build, the 200MB models 
# are baked directly into the image. When your server boots, 
# it will start instantly instead of hanging to download files.
# ---------------------------------------------------------
RUN uv run python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding(model_name='Qdrant/bm25')"

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
