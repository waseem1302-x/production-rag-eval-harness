FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Separate COPYs: a multi-source COPY flattens directory contents
# under Buildah (verified 18 Nov 2026: `COPY a/ b/ ./` put
# scripts/build_index.py at ./build_index.py, so `python -m scripts`
# failed). One COPY per directory is unambiguous in Docker and Buildah.
COPY requirements.txt .
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY eval/ ./eval/

RUN pip install --no-cache-dir -r requirements.txt

# Build the chunk + FAISS index at image build time so the container
# starts without re-fetching the corpus or the embedding model.
# NOTE: this step downloads the embedding model from Hugging Face, so
# the build needs network access. In this VM the egress proxy
# terminates TLS, so the build also needs the proxy CA bundle mounted
# (buildah: -v /run/hatch/egress-tls/ca-bundle.pem:/etc/ssl/certs/ca-certificates.crt:ro);
# on a normal machine with Docker no extra flag is needed.
RUN python -m scripts.build_index

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
