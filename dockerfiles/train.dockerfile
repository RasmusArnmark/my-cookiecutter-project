# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set up Python dependencies with caching
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=~/.cache/pip pip install -r requirements.txt

# Copy project files
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ /data/
COPY models/ /models/
COPY reports/ /reports/

# Set working directory and entrypoint
WORKDIR /
ENTRYPOINT ["python", "-u", "src/cookiecutter1/train.py"]
