# Vev server, with the pinned versions of requirements-serve.txt (docs/serving.md). Weights are not in the image; they
# are downloaded from Hugging Face on first start (mount a volume at /root/.cache/huggingface to keep them, and one at
# /root/.triton to keep the compiled kernels).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Triton compiles the flash-linear-attention kernels at run time and needs a C compiler; the runtime image has none.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir --no-deps -r requirements-serve.txt
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY vev ./vev
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps . && chmod +x scripts/serve_board.sh

EXPOSE 8009
ENTRYPOINT ["vev", "serve", "--host", "0.0.0.0", "--port", "8009"]
CMD ["--model", "CountingSheep/vev-4b-v2"]
