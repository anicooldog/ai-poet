from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
# from dotenv import load_dotenv
import os

# .env 파일에서 환경변수 불러오기
# load_dotenv()

# LLM 설정 (Gemini Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# LangSmith 트래커 설정
tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))
llm_with_tracing = llm.with_config({"callbacks": [tracer]})

# prompt trmplate 생성
prompt = ChatPromptTemplate.from_messages([
    ("system","You are helpful assistant."),
    ("user","{input}")
])

# 문자열 출력 파서
output_parser = StrOutputParser()


# LLM 체인 구성 
chain = prompt | llm_with_tracing | output_parser

# 제목
st.title("인공지능 시인")

# 시 주제 입력 필드
content = st.text_input("시의 주제를 제시해 주세요")
st.write("시의 주제는", content)


# 시 작성 요청하기
if st.button("시 작성 요청하기"):
 with st.spinner('Wait for it...'):
    result = chain.invoke({"input": content + "에 대한 시를 써줘"})
    st.write(result)
