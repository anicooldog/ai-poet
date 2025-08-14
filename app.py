# app.py
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# 환경 변수 로드
# -----------------------------
# 기본 로드
load_dotenv()
# 보조 탐색 (원하시는 경로 추가 가능)
candidates = [
    Path(".env"),
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent / "poet" / ".env",
]
for p in candidates:
    if p.exists():
        load_dotenv(dotenv_path=p, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.title("📄 ChatPDF (Upload → Split → Embed → Ask)")
st.write("---")

# 파일 업로드
upload_file = st.file_uploader("pdf 파일을 올려주세요!", type=["pdf"])
st.write("---")

# 세션 상태 키 초기화
for k in ["texts", "vectorstore", "retriever_ready"]:
    if k not in st.session_state:
        st.session_state[k] = None

# -----------------------------
# 유틸: PDF -> Documents
# -----------------------------
def pdf_to_documents(upload):
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, upload.name)
    with open(temp_path, "wb") as f:
        f.write(upload.getvalue())
    loader = PyPDFLoader(temp_path)
    # load()는 페이지 단위 Document 리스트 반환
    return loader.load()  # temp_dir는 함수 끝나도 파일은 열람 가능(로드됨)

# -----------------------------
# 단계 1) 업로드 후 분할
# -----------------------------
if upload_file is not None:
    try:
        pages = pdf_to_documents(upload_file)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = splitter.split_documents(pages)
        st.session_state.texts = texts
        st.info(f"✅ 텍스트 분할 완료: 총 {len(texts)}개 청크")

    except Exception as e:
        st.error(f"PDF 처리 중 오류: {type(e).__name__}: {e}")
        st.stop()

# -----------------------------
# 단계 2) 임베딩 & 벡터DB 구축
# -----------------------------
if st.session_state.texts is not None:
    if st.session_state.vectorstore is None:
        if st.button("벡터화(임베딩) 시작"):
            if not GOOGLE_API_KEY:
                st.error("GOOGLE_API_KEY가 설정되지 않았습니다. .env 또는 환경변수를 확인하세요.")
                st.stop()
            try:
                embeddings = HuggingFaceEmbeddings(model_name="upskyy/bge-m3-korean")
                
                # 메모리 벡터DB (필요 시 persist_directory 지정)
                st.session_state.vectorstore = Chroma.from_documents(
                    st.session_state.texts, embeddings
                )
                st.success("✅ 임베딩 및 벡터DB 구축 완료")
            except Exception as e:
                st.error(f"임베딩 단계에서 오류: {type(e).__name__}: {e}")

# -----------------------------
# 단계 3) RAG 질의
# -----------------------------
if st.session_state.vectorstore is not None:
    st.header("PDF에게 질문해 보세요!!")
    question = st.text_input("질문을 입력하세요", placeholder="예) 이 문서에서 설치 절차를 요약해줘")

    # LLM 준비 (질의 시점에 생성)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        # temperature=0.2  # 원하시면 보수적으로 조정
    )

    # MultiQueryRetriever: 질문을 다양한 관점으로 확장해 더 좋은 문서를 검색
    base_retriever = st.session_state.vectorstore.as_retriever()

    # 프롬프트 (로컬 템플릿 사용: hub.pull 의존 제거)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "당신은 신중하고 사실에 근거한 도우미입니다.\n"
            "다음 컨텍스트만 근거로 질문에 한국어로 답하세요. "
            "컨텍스트에 없는 내용은 추측하지 말고 '모르겠습니다'라고 답하세요.\n\n"
            "컨텍스트:\n{context}\n\n"
            "질문:\n{question}\n\n"
            "답변:"
        ),
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    if st.button("질문하기"):
        if not question.strip():
            st.warning("질문을 입력해 주세요.")
            st.stop()

        try:
            mq_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )

            # RAG 체인
            rag_chain = (
                {
                    "context": mq_retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.spinner("답변 생성 중..."):
                answer = rag_chain.invoke(question)

            st.subheader("답변")
            st.write(answer)

            # 참고로 사용된 문서 조각도 보여주기 (상위 3개)
            with st.expander("🔎 사용된 컨텍스트(상위 3개 보기)"):
                ctx_docs = base_retriever.get_relevant_documents(question)[:3]
                for i, d in enumerate(ctx_docs, 1):
                    st.markdown(f"**[컨텍스트 {i}]**")
                    st.write(d.page_content[:1500] + ("..." if len(d.page_content) > 1500 else ""))
                    meta = d.metadata or {}
                    src = f"page={meta.get('page', 'N/A')}, source={meta.get('source', 'N/A')}"
                    st.caption(src)

        except Exception as e:
            st.error(f"질의 처리 중 오류: {type(e).__name__}: {e}")

# -----------------------------
# 안내
# -----------------------------
st.write("---")
st.caption(
    "Tip: 임베딩은 버튼을 눌렀을 때 한 번만 수행되며, 같은 세션에서는 재사용됩니다. "
    "오류가 계속되면 패키지를 업데이트해 보세요: "
    "`pip install -U langchain langchain-community langchain-text-splitters "
    "langchain-google-genai google-generativeai chromadb`"
)