# -- coding: utf-8 --
# 豆包 Seed 1.6 图片理解示例
import importlib
import os
from dotenv import load_dotenv

load_dotenv()

# seed1.6.py 文件名含句点，无法直接 import，需用 importlib
seed16 = importlib.import_module("doubao.seed1.6")

if __name__ == "__main__":
    img_path = "your_image.png"  # 替换为你的图片路径
    prompt = "描述这张图片的内容"  # 替换为你的提示词

    images_base64, mime_type = seed16.url_to_base64(img_path)
    result = seed16.ask_doubao_images(prompt, images_base64, mime_type)
    print(result)
