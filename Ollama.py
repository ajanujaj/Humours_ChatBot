from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

few_shot_examples = """
Human: What's the capital of France?
assistant: The capital of France is Paris.

Human: Who wrote 'To Kill a Mockingbird'?
assistant: 'To Kill a Mockingbird' was written by Harper Lee.

Human: Can you tell me a joke?
assistant: Sure! Why don't scientists trust atoms? Because they make up everything!
"""


prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=f"""You are an AI agent, be friendly and have some sense of humour.

    {few_shot_examples}
    
    chat_history: {{chat_history}}

    Human: {{question}}
    
    assistant:"""
)

llm = Ollama(
    model="llama2",
    temperature=0.7,
    # max_tokens=150,
    top_p=0.9,
    stop=["Human:"],
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5)

llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

st.set_page_config(
    page_title="Humours-chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Humours-chat")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, how can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)