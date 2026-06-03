# LLM Power

LLM/AI 能力集成工具集，封装各平台 API，方便在项目中调用。

## 模块

| 模块 | 说明 |
|------|------|
| `qwen/` | 通义千问 API（OpenAI 兼容接口） |
| `doubao/seed1.6.py` | 豆包 Seed 1.6 图片理解 |
| `doubao/realtime_dialog/` | 豆包实时语音对话（WebSocket） |
| `video_gen/runway_api.py` | Runway 图生视频 |
| `examples/` | 各模块使用示例 |

## 快速开始

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入各平台密钥

# 2. 安装依赖（按需）
pip install -r qwen/requirements.txt
pip install -r doubao/realtime_dialog/requirements.txt
pip install -r video_gen/requirements.txt
```

## 示例

```bash
python examples/qwen_test.py
python examples/doubao_seed1.6_test.py
```

## 环境变量

| 变量 | 用途 |
|------|------|
| `QWEN_API_KEY` / `QWEN_BASE_URL` | 通义千问 |
| `RUNWAYML_API_SECRET` | Runway |
| `DOUBAO_APP_ID` / `DOUBAO_ACCESS_KEY` / `DOUBAO_APP_KEY` | 豆包实时语音 |
| `DOUBAO_SEED_API_KEY` / `DOUBAO_SEED_URL` | 豆包 Seed 1.6 |
