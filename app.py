import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

st.title("Chat with Wikipedia and Arxiv")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key: ", type="password")
if api_key:
    st.sidebar.success("API key set successfully.")

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

search = DuckDuckGoSearchRun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am a chatbot assistant who have ability to use Wikipedia and Arxiv. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768")
    tools = [arxiv, wikipedia, search]

    agent = initialize_agent(
        llm=llm,
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)