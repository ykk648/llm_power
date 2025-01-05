# -- coding: utf-8 --
# @Time : 2025/1/5
# @Author : ykk648
import time
import os
from runwayml import RunwayML
from dotenv import load_dotenv

load_dotenv()

RUNWAYML_API_SECRET = os.getenv("RUNWAYML_API_SECRET")

client = RunwayML(api_key=RUNWAYML_API_SECRET)


def gen_video(prompt_image, prompt_text, duration=5, ratio="9:16"):
    task = client.image_to_video.create(
        model='gen3a_turbo',
        prompt_image=prompt_image,
        prompt_text=prompt_text,
        duration=duration,
        ratio=ratio
    )
    task_id = task.id

    # Poll the task until it's complete
    time.sleep(10)  # Wait for a second before polling
    task = client.tasks.retrieve(task_id)
    while task.status not in ['SUCCEEDED', 'FAILED']:
        time.sleep(10)  # Wait for ten seconds before polling
        task = client.tasks.retrieve(task_id)
    return task


if __name__ == '__main__':
    try:
        task = gen_video(prompt_image='image_url',
                         prompt_text="prompt_text", duration=5, ratio="768:1280")
        print(task.output[0])
    except Exception as e:
        print(e)
