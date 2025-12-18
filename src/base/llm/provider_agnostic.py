import requests
import json
from enum import Enum
from typing import List, Dict, Optional, Any, Generator

from openai import OpenAI


class ProviderAgnosticLLM:
    """
    Provider-agnostic LLM client supporting OpenAI, Gemini etc.
    Swap provider by changing api_url, api_key, and the model.
    """

    class Providers(Enum):
        OPENAI = "https://api.openai.com/v1/chat/completions"
        GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
        DEEPSEEK = "https://api.deepseek.com/v1"
        GROQ = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        provider: Providers,
        api_key: str,
        model: str,
        system_prompt: str = "",
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt

        self.client = OpenAI(
            api_key=self.api_key, base_url=self.provider.value
        )

        # Initialize chat history
        self.chat_history: List[Dict[str, str]] = []
        if self.system_prompt:
            self.add_message(role="system", content=self.system_prompt)

    def add_message(self, role: str, content: Any) -> None:
        self.chat_history.append(dict(role=role, content=content))

    def chat(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
    ) -> str:
        """
        Send a non-streaming request to OpenAI's Chat API.
        """
        all_messages = self.chat_history.copy()
        if all_messages:
            all_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        # Extract the assistant's reply
        reply = response["choices"][0]["message"]["content"]

        # Update chat history
        if all_messages:
            self.add_message(role="user", content=all_messages[-1]["content"])
        self.add_message(role="assistant", content=reply)

        return reply

    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
    ) -> Generator[str, None, None]:
        """
        Stream response from OpenAI's Chat API.
        """
        all_messages = self.chat_history.copy()
        if all_messages:
            all_messages.extend(messages)

        # OpenAI's streaming API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
        )

        buffer = ""
        for chunk in response:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                chunk_content = delta.get("content", "")
                if chunk_content:
                    buffer += chunk_content
                    yield chunk_content

        # Save to history after streaming
        if all_messages:
            self.chat_history.append(
                {"role": "user", "content": all_messages[-1]["content"]}
            )
        self.chat_history.append({"role": "assistant", "content": buffer})

    def send_user_message(self, text: str, stream: bool = False) -> Any:
        user_message = [{"role": "user", "content": text}]
        return self.stream(user_message) if stream else self.chat(user_message)

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        if self.chat_history and self.chat_history[0]["role"] == "system":
            self.chat_history[0]["content"] = prompt
        else:
            self.chat_history.insert(0, {"role": "system", "content": prompt})

    def clear_chat(self):
        self.chat_history = []
        if self.system_prompt:
            self.add_message(role="system", content=self.system_prompt)
