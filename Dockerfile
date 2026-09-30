# Vev server. Weights are not in the image; they are downloaded from Hugging Face on first start
# (mount a volume at /root/.cache/huggingface to keep them).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY vev ./vev
RUN pip install --no-cache-dir .

EXPOSE 8009
ENTRYPOINT ["vev", "serve", "--host", "0.0.0.0", "--port", "8009"]
CMD ["--model", "CountingSheep/vev-4b"]
