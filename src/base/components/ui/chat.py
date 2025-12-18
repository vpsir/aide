from typing import Any

import streamlit as st

from src.base.llm.provider_agnostic import ProviderAgnosticLLM


class ChatUI:
    def __init__(
        self,
        title: str,
        caption: str,
        llm: ProviderAgnosticLLM,
        initial_message: str = "",
        system_prompt: str = "",
    ):
        self.llm = llm
        self.title = title
        self.caption = caption
        self.initial_message = initial_message
        self.system_prompt = system_prompt

        # Component identifier
        self.cid = self.__class__.__name__

        # syncing the chat history
        st.session_state[f"{self.cid}_messages"] = self.llm.chat_history.copy()

    @property
    def _messages(self) -> list:
        return st.session_state[f"{self.cid}_messages"]

    def _add_message(self, role: str, message: Any) -> None:
        st.session_state[f"{self.cid}_messages"].append(
            {"role": role, "content": message}
        )
        self._render_messages()

    def _render_messages(self) -> None:
        for message in self._messages:
            st.chat_message(message["role"]).write(message["content"])

    def _chat(self) -> None:
        user_prompt = st.chat_input()
        if user_prompt:
            # Add user message
            self._add_message("user", user_prompt)

            # Stream assistant response
            assistant_content = ""
            self._add_message("assistant", "")  # placeholder
            for chunk in self.llm.send_user_message(user_prompt, stream=True):
                assistant_content += chunk
                st.session_state[self.cid][-1]["content"] = assistant_content
                st.empty()
                self.display()

    def display(self) -> None:
        st.title(self.title)
        st.caption(self.caption)

        # Initialize session state for this component
        if f"{self.cid}_messages" not in st.session_state:
            st.session_state[f"{self.cid}_messages"] = []
            if self.system_prompt:
                self._add_message(role="system", message=self.system_prompt)
            if self.initial_message:
                self._add_message(
                    role="assistant", message=self.initial_message
                )

        # Start of chatting!
        self._chat()
