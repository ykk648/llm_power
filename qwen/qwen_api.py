# -- coding: utf-8 --
# @Time : 2024/11/17
# @Author : ykk648
"""
ref https://github.com/Henry-23/VideoChat/blob/master/src/llm.py
rewrite by Claude
"""
from openai import OpenAI
import re
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QwenAPI:
    """QwenAPI class for interacting with Qwen language models through OpenAI-compatible API.
    
    Attributes:
        model (str): The Qwen model identifier to use for completions
        client (OpenAI): OpenAI client instance for API communication
    """

    def __init__(self, api_key=None, base_url=None):
        """Initialize the QwenAPI.
        
        Args:
            api_key (str, optional): API key for authentication. Defaults to env QWEN_API_KEY.
            base_url (str, optional): Base URL for API endpoint. Defaults to env QWEN_BASE_URL.
        """
        self.model = "Qwen/Qwen2.5-1.5B-Instruct"
        self.prompt = 'qwen/prompt.txt'

        # Get configuration from environment variables if not provided
        api_key = api_key or os.getenv("QWEN_API_KEY")
        base_url = base_url or os.getenv("QWEN_BASE_URL")

        if not api_key or not base_url:
            raise ValueError("API key and base URL must be provided either through parameters or environment variables")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def infer(self, user_input, user_messages):
        """Generate a response for a single message.
        
        Args:
            user_input (str): The user's input message
            user_messages (list): List of previous messages in the conversation.
                Each message should be a dict with 'role' and 'content' keys.
        
        Returns:
            tuple: (str, list) - The model's response and updated message history
        """
        # prompt
        if len(user_messages) == 1:
            with open(self.prompt, 'r', encoding='utf-8') as f:
                user_messages[0]['content'] = f.read()
        user_messages.append({'role': 'user', 'content': user_input})
        print(user_messages)

        try:
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=user_messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=512,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            chat_response = chat_response.choices[0].message.content.strip()
            user_messages.append({'role': 'assistant', 'content': chat_response})

            if len(user_messages) > 10:
                user_messages.pop(0)

            print(f'[Qwen API] {chat_response}')
            return chat_response, user_messages
        except Exception as e:
            print(f"调用通义千问API时发生错误: {e}")
            return "抱歉，我现在无法回答，请稍后再试。", user_messages

    def infer_stream(self, user_input, user_messages, llm_queue, chunk_size):
        """Generate a streaming response for a message.
        
        Args:
            user_input (str): The user's input message
            user_messages (list): List of previous messages in the conversation.
                Each message should be a dict with 'role' and 'content' keys.
            llm_queue (Queue): Queue for receiving streamed response chunks
            chunk_size (int): Size of text chunks to accumulate before sending to queue
        
        Returns:
            tuple: (str, list, list) - The complete response, updated message history, and time costs
        """
        print(f"[LLM] User input: {user_input}")
        time_cost = []
        start_time = time.time()
        # prompt
        if len(user_messages) == 1:
            with open(self.prompt, 'r', encoding='utf-8') as f:
                user_messages[0]['content'] = f.read()
        print(f"[LLM] user_messages: {user_messages}")
        user_messages.append({'role': 'user', 'content': user_input})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=user_messages,
            stream=True
        )

        chat_response = ""
        buffer = ""
        sentence_buffer = ""
        sentence_split_pattern = re.compile(r'(?<=[,;.!?，；：。:！？》、”])')
        fp_flag = True
        print("[LLM] Start LLM streaming...")
        for chunk in completion:
            chat_response_chunk = chunk.choices[0].delta.content
            chat_response += chat_response_chunk
            buffer += chat_response_chunk

            sentences = sentence_split_pattern.split(buffer)

            if not sentences:
                continue

            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                sentence_buffer += sentence

                if fp_flag or len(sentence_buffer) >= chunk_size:
                    llm_queue.put(sentence_buffer)
                    time_cost.append(round(time.time() - start_time, 2))
                    start_time = time.time()
                    print(f"[LLM] Put into queue: {sentence_buffer}")
                    sentence_buffer = ""
                    fp_flag = False

            buffer = sentences[-1].strip()

        sentence_buffer += buffer
        if sentence_buffer:
            llm_queue.put(sentence_buffer)
            print(f"[LLM] Put into queue: {sentence_buffer}")

        llm_queue.put(None)

        user_messages.append({'role': 'assistant', 'content': chat_response})
        if len(user_messages) > 10:
            user_messages.pop(0)

        print(f"[LLM] Response: {chat_response}\n")

        return chat_response, user_messages, time_cost
