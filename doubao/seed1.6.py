import base64
import json
import os
from pathlib import Path

import imghdr
import requests
from dotenv import load_dotenv

load_dotenv()

DOUBAO_URL = os.getenv("DOUBAO_SEED_URL", "http://oneapi-test.cds8.cn/v1/chat/completions")
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.getenv("DOUBAO_SEED_API_KEY", "")}'
}
COMPONENT_NAME = 'Doubao-Seed-1.6'


def url_to_base64(url: str) -> str:
    path = Path(url)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    # 检测图像格式（仅支持JPEG/PNG）
    img_type = imghdr.what(url)
    if img_type not in ('jpeg', 'png'):
        raise ValueError(f"不支持的图像格式: {img_type}。仅支持JPEG/PNG")

    # 读取并编码
    with open(path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    # 构建请求
    mime_type = "image/jpeg" if img_type == 'jpeg' else "image/png"
    return image_base64, mime_type


def ask_doubao_images(prompt, image_base64, mime_type):
    url_content = [{
        "type": "image_url",
        "image_url": {
            # "url": f"data:image/png;base64, {base64_string}",
            "url": f"data:{mime_type};base64,{image_base64}"
        }
    }]
    data = {
        "model": COMPONENT_NAME,
        "messages": [
            {
                "role":
                    "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }] + url_content
            }],
        "thinking": {
            "type": "disabled"  # 不使用深度思考能力,
            # "type": "enabled" # 使用深度思考能力
            # "type": "auto" # 模型自行判断是否使用深度思考能力
        }
    }
    response = requests.post(DOUBAO_URL,
                             headers=HEADERS,
                             data=json.dumps(data), timeout=60)
    answer = response.json()['choices'][0]['message']['content']

    return answer
