# -- coding: utf-8 --
# @Time : 2024/12/6
# @Author : ykk648

from qwen.qwen_api import QwenAPI
from queue import Queue
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_normal_chat():
    """Test normal chat completion"""
    api_key = os.environ.get('QWEN_API_KEY')
    base_url = os.environ.get('QWEN_BASE_URL')
    qwen = QwenAPI(api_key=api_key, base_url=base_url)

    # Initialize conversation history
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    ]

    print("\n=== Starting normal chat test ===")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            break

        response, messages = qwen.infer(user_input, messages)
        print(f"\nQwen: {response}")


def test_streaming_chat():
    """Test streaming chat completion"""
    api_key = os.environ.get('QWEN_API_KEY')
    base_url = os.environ.get('QWEN_BASE_URL')
    qwen = QwenAPI(api_key=api_key, base_url=base_url)

    # Initialize conversation history and queue
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    ]
    queue = Queue()

    print("\n=== Starting streaming chat test ===")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            break

        # Start streaming response
        response, messages, time_costs = qwen.infer_stream(user_input, messages, queue, chunk_size=10)
        print(f"\nFinal response: {response}")
        print(f"Time costs: {time_costs}")


if __name__ == '__main__':
    # Choose which test to run
    test_type = input("Choose test type (1: Normal, 2: Streaming): ").strip()

    if test_type == "1":
        test_normal_chat()
    elif test_type == "2":
        test_streaming_chat()
    else:
        print("Invalid choice. Please choose 1 or 2.")
