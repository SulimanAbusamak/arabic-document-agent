# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Where sentence-transformers caches the embedding model. Set before the
# install so the model downloaded at BUILD time lands here and ships inside
# the image -- the container then starts with no network call and no wait.
ENV HF_HOME=/opt/hf \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU-only torch, installed BEFORE requirements.txt on purpose.
#
# sentence-transformers depends on torch. Left alone, pip takes the default
# PyPI wheel, which on Linux bundles the CUDA libraries -- about 2.5 GB of
# GPU code for a machine with no GPU. Installing the CPU build first means
# torch is already satisfied when sentence-transformers is resolved, so pip
# leaves it alone. Roughly 250 MB instead of 2.5 GB, same behaviour here.
RUN pip install --no-cache-dir \
        torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image.
#
# Without this the first container to start downloads ~470 MB from
# HuggingFace, which fails on an offline host and makes the first invoice
# processed mysteriously slow. Downloading at build time means a bigger
# image and a predictable runtime -- the right trade for something that
# runs unattended.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY src/ ./src/

# Written to at runtime, so they exist as mount points.
RUN mkdir -p data cache data/uploads data/chroma

EXPOSE 8501

# Streamlit binds to localhost by default, which inside a container means
# "reachable only from inside this container". 0.0.0.0 is what publishes it.
CMD ["streamlit", "run", "src/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]