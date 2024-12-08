### Deploy by vllm

#### Server

```shell
# install torch==2.5.1
pip install vllm
# hf proxy
export HF_ENDPOINT=https://hf-mirror.com
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0
```

#### Client

ref `qwen_api.py`

### Deploy by modelscope

ref `qwen_local.py`