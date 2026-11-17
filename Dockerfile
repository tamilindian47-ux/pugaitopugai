FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      torch==2.9.1 torchvision==0.24.1 \
      --index-url https://download.pytorch.org/whl/cu128

RUN printf 'torch==2.9.1\ntorchvision==0.24.1\n' \
    > /app/constraints.txt

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir \
    -c /app/constraints.txt -r requirements.txt

RUN python3 -m pip check

RUN python3 -c "\
import inspect, torch, torchvision; \
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, GGUFQuantizationConfig; \
assert hasattr(torch, 'accelerator'); \
assert hasattr(torchvision.ops, 'nms'); \
params = inspect.signature(QwenImage21Pipeline.__call__).parameters; \
assert 'image' in params and 'true_cfg_scale' in params; \
print('Dependency imports and pipeline interface checks passed:', torch.__version__, torchvision.__version__)"

COPY handler.py .

RUN python3 -m py_compile handler.py

CMD ["python3", "-u", "handler.py"]
