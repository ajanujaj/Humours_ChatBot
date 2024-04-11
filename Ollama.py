from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["chat_history","question"],
    template ="""You are AI agent, be freindly and have some sense of humour.

        chat_history: {chat_history}

        Human: {question}
        
        assistant:"""

)

llm = Ollama(
    model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

memory = ConversationBufferWindowMemory(memory_key="chat_history",k=5)

llm_chain =LLMChain(
    llm=llm,
    memory =memory,
    prompt = prompt
)
# llm_chain("The first man on the summit of Mount Everest, the highest peak on Earth, was ...")

st.set_page_config(
    page_title="SignLanguage-GPT",
    page_icon="🤖",
    layout="wide"
)

st.title("SignLanguage-GPT")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant","content":"Hello there, how can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append(
        {"role":"user","content":user_prompt}
    )
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    
    new_ai_message =  {"role":"assistant","content":ai_response}
    print(new_ai_message)
    st.session_state.messages.append(new_ai_message)


