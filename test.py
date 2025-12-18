# chat_ui.py
import streamlit as st
from typing import List, Dict
from llm_client import ProviderAgnosticLLM


def render_bubble(role: str, content: str):
    """Render LinkedIn-style bubble in Streamlit."""
    align = "flex-end" if role == "user" else "flex-start"
    bg = "#0A66C2" if role == "user" else "#F3F2EF"
    color = "white" if role == "user" else "black"

    st.markdown(
        f"""
        <div style="display: flex; justify-content: {align}; margin: 4px 0;">
            <div style="
                background-color: {bg};
                color: {color};
                padding: 12px 16px;
                border-radius: 14px;
                max-width: 70%;
                font-size: 15px;
                line-height: 1.4;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen;
                box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
                white-space: pre-wrap;
            ">
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


class LinkedInChatUI:
    """Streamlit chat UI integrated with any LLM."""

    def __init__(self, llm_client: ProviderAgnosticLLM):
        self.llm = llm_client
        self.cid = "LinkedInChatUI"
        if self.cid not in st.session_state:
            st.session_state[self.cid] = self.llm.chat_history.copy()

    def display_messages(self):
        for msg in st.session_state[self.cid]:
            render_bubble(msg["role"], msg["content"])

    def add_message(self, role: str, content: str):
        st.session_state[self.cid].append({"role": role, "content": content})

    def display(self):
        st.title("LinkedIn Assistant")
        st.caption("Your friendly LinkedIn-style chat assistant")
        self.display_messages()

        user_input = st.chat_input("Type a message...")
        if user_input:
            self.add_message("user", user_input)
            render_bubble("user", user_input)

            # Stream assistant response
            assistant_content = ""
            self.add_message("assistant", "")  # placeholder
            for chunk in self.llm.send_user_message(user_input, stream=True):
                assistant_content += chunk
                st.session_state[self.cid][-1]["content"] = assistant_content
                st.empty()
                self.display_messages()


# ----------------------------------------------------------------------------


import streamlit as st
from llm_client import ProviderAgnosticLLM
from chat_ui import LinkedInChatUI

# Change api_url & api_key to switch providers
llm = ProviderAgnosticLLM(
    api_key="YOUR_API_KEY",
    api_url="https://api.openai.com/v1/chat/completions",
    model="gpt-4",
    system_prompt="You are a helpful LinkedIn assistant.",
)

chat_ui = LinkedInChatUI(llm)
chat_ui.display()
