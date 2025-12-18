import streamlit as st

from src.ui.linkedin.display import LinkedInChatUI


def main():
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key", key="chatbot_api_key", type="password"
        )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    linkedin_ui = LinkedInChatUI()
    linkedin_ui.display()


if __name__ == "__main__":
    main()
