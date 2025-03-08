"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import faiss
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.faiss import FAISS
# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.embeddings import HuggingFaceEmbeddings
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

icon = "Tobig2.png"
profile = Image.open('Tobig2.png')
profile2 = Image.open('Tobig17th.png')
st.set_page_config(page_title="Tobig2 QA Bot", page_icon=icon)
##################################################################################
# 사이드바
with st.sidebar:
    choose = option_menu("Tobigs", ["About", "Codebot", "Conference bot"],
                         icons=['house', 'robot','archive-fill'], 
                         menu_icon="menu-button", default_index=0,
                         styles={
                                "container": {"padding": "5!important", "background-color": "#fafafa"},
                                "icon": {"color": "gray", "font-size": "25px"}, 
                                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "#02ab21"},
                         }
    )
##################################################################################
# 파트별 컨테이너화
header_container = st.container()
codebot_container = st.container()
conference_container = st.container()
##################################################################################
# About 페이지
if choose == "About":
    with header_container:
        st.header(":blue[Welcome to 17th Conference]")
        st.image(profile)
        
        st.markdown(
        """
        빅데이터를 공부하고 큰 사람이 되기 위해서 모인 사람들, 투빅스를 소개합니다

        [🎆 소개 링크 CLICK ◀](https://www.notion.so/CLICK-e74a26737a8c42e0965e1520b7f6d768?pvs=21)

        [📚 커리큘럼 링크 CLICK ◀](https://www.notion.so/CLICK-9b2b34483f0f4a3e86d60637d6b2806b?pvs=21)

        [🗣 컨퍼런스 링크 CLICK ◀](https://www.notion.so/CLICK-d48f8fab2154479aa257b5f7e5fe5b00?pvs=21)
        """)
        st.image(profile2)
        ## 투빅스 공식 카카오톡 채널
        st.markdown(
        """
        ---

        [📢 카카오톡](http://pf.kakao.com/_QyxiDxb)

        (위 링크에서 '**채널 추가**' 버튼을 누르시면 투빅이를 만나보실 수 있습니다)

        카카오톡에서 **tobigs** 검색, '채널추가' 후 '채팅하기' 버튼을 누르시면 **공식 Q&A**를 확인하실 수 있습니다. 
    """)
##################################################################################
# Codebot 페이지
elif choose == "Codebot":
    with codebot_container:
        index = faiss.read_index("docs.index")

        with open("faiss_store.pkl", "rb") as f:
            store = pickle.load(f)

        store.index = index

        def get_conversation_chain(vetorestore):
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            template = """
            You are an AI assistant for answering questions with code.
            Provide a conversational answer in korean except code line.

            
            {context}
            Question: {question}
            """
            QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=vetorestore.as_retriever(search_type = 'similarity', vervose = True), 
                    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
                    get_chat_history=lambda h: h,
                    return_source_documents=True,
                    verbose = True
                )
            conversation_chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
            return conversation_chain
            
        st.title(":blue[Tobig2 Codebot]")
        # From here down is all the StreamLit UI.
        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        if "processComplete" not in st.session_state:
            st.session_state.processComplete = None

        st.session_state.conversation = get_conversation_chain(store) 
        st.session_state.processComplete = True

        if 'messages' not in st.session_state:
            st.session_state['messages'] = [{"role": "assistant", 
                                            "content": "안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
        # Display chat messages from history on app rerun
        # Custom avatar for the assistant, default avatar for user
        for message in st.session_state.messages:
            if message["role"] == 'assistant':
                with st.chat_message(message["role"], avatar=icon):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
        history = StreamlitChatMessageHistory(key="chat_messages")
        # Chat logic
        if query := st.chat_input("질문을 입력해주세요."):
            # 이전 대화 내용 클리어
            st.session_state.messages.clear()

            # 새 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant", avatar=icon):
                chain = st.session_state.conversation
                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']
                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)

            # 새 AI 응답 메시지 추가
            st.session_state.messages.append({"role": "assistant", "content": response})

##################################################################################
# Conference bot 페이지
elif choose == "Conference bot":
    with conference_container:
        index = faiss.read_index("docs_confernce.index")   ## FAISS에 저장된 내용을 가져옴

        with open("faiss_store_confernce.pkl", "rb") as f:
            store = pickle.load(f)

        store.index = index

        def get_conversation_chain(vetorestore):
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
            template = """
            You are an AI assistant for answering conference questions.
            Provide a conversational answer in korean.
            
            {context}
            Question: {question}
            """
            QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=vetorestore.as_retriever(search_type = 'similarity', vervose = True,search_kwargs={"k": 5}), 
                    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
                    get_chat_history=lambda h: h,
                    return_source_documents=True,
                    verbose = True
                )
            conversation_chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
            return conversation_chain
            
        st.title(":blue[Tobig2 Conference bot]")
        # From here down is all the StreamLit UI.
        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        if "processComplete" not in st.session_state:
            st.session_state.processComplete = None

        st.session_state.conversation = get_conversation_chain(store) 
        st.session_state.processComplete = True

        if 'messages' not in st.session_state:
            st.session_state['messages'] = [{"role": "assistant", 
                                            "content": "안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
        # Display chat messages from history on app rerun
        # Custom avatar for the assistant, default avatar for user
        for message in st.session_state.messages:
            if message["role"] == 'assistant':
                with st.chat_message(message["role"], avatar=icon):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
        history = StreamlitChatMessageHistory(key="chat_messages")
        # Chat logic
        if query := st.chat_input("질문을 입력해주세요."):
            # 이전 대화 내용 클리어
            st.session_state.messages.clear()

            # 새 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant", avatar=icon):
                chain = st.session_state.conversation
                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']
                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)

            # 새 AI 응답 메시지 추가
            st.session_state.messages.append({"role": "assistant", "content": response})
    