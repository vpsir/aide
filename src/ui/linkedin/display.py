import streamlit as st

from src.base.llm.provider_agnostic import ProviderAgnosticLLM
from src.base.components.ui.chat import ChatUI


class LinkedInChatUI(ChatUI):
    def __init__(self):
        super().__init__(
            title="LinkedIn Assistant",
            caption="Some fancy caption for LinkedIn Assistant",
            llm=ProviderAgnosticLLM(
                provider=ProviderAgnosticLLM.Providers.GEMINI,
                api_key="AIzaSyARJsKgOwf_r5LI0U6U0q89tIAyg0JjZ6Q",
                model="gemini-2.5-flash",
                system_prompt="You're LinkedIn Expert.",
            ),
            initial_message="I can make optimize your linkedIn profile.",
        )
