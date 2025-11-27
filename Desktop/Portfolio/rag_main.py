import os
from dotenv import load_dotenv

# 시작 시 conda activate rag_project
# 처음엔 단일 파일로 기능 구현에 집중했고, 이후 유지보수를 위해 모듈화를 진행
# 1. 환경변수 로드 (GOOGLE_API_KEY 가져오기)

load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))

# 라이브러리 임포트 (Google용으로 변경됨)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # 여기가 핵심 변경!
from langchain_community.vectorstores import Chroma

def ingest_docs():
    # ---------------------------------------------------------
    # 단계 1: 문서 로드 (Load)
    # 폴더에 있는 PDF 파일명과 일치시켜주세요.
    # ---------------------------------------------------------
    pdf_filename = "report.pdf" 
    
    if not os.path.exists(pdf_filename):
        print(f"❌ 오류: '{pdf_filename}' 파일이 없습니다. 폴더에 PDF를 넣어주세요.")
        return

    print(f"📂 1. '{pdf_filename}' 문서를 불러오는 중...")
    loader = PyPDFLoader(pdf_filename)
    raw_documents = loader.load()
    print(f"   👉 총 {len(raw_documents)} 페이지를 로드했습니다.")

    # ---------------------------------------------------------
    # 단계 2: 문서 분할 (Split)
    # ---------------------------------------------------------
    print("✂️ 2. 문서를 작은 조각(Chunk)으로 자르는 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"   👉 총 {len(documents)} 개의 청크로 분할되었습니다.")

    # ---------------------------------------------------------
    # 단계 3 & 4: 임베딩 생성 및 DB 저장 (Embed & Store)
    # Google의 최신 임베딩 모델인 'models/text-embedding-004'를 사용합니다. (무료)
    # ---------------------------------------------------------
    print("💾 3. Google Gemini 임베딩을 사용하여 Vector DB에 저장 중...")
    api_key = os.getenv("GOOGLE_API_KEY")

    # 키를 함수 안에 직접 넣어줍니다 (google_api_key 파라미터)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key 
    )
    
    # DB 저장 (./chroma_db 폴더에 저장)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    
    print("✅ 저장 완료! './chroma_db' 폴더가 생성되었습니다.")



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

def ask_bot():      
    print("🤖 챗봇을 로딩하고 있습니다... 잠시만 기다려주세요.")
    
    # .env 로드 확인
    api_key = os.getenv("GOOGLE_API_KEY")

    # ---------------------------------------------------------
    # [설정 1] 임베딩 모델 준비 (DB를 읽으려면 1일차와 똑같은 모델 필요)
    # ---------------------------------------------------------
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key
    )

    # ---------------------------------------------------------
    # [설정 2] 저장해둔 기억(DB) 불러오기
    # ---------------------------------------------------------
    if not os.path.exists("./chroma_db"):
        print("❌ 오류: 'chroma_db' 폴더가 없습니다. 먼저 ingest_docs()를 실행하세요.")
        return

    # DB 로드 (저장된 폴더 경로 지정)
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    # 검색기(Retriever) 설정: 가장 유사한 문서 3개를 가져옴
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ---------------------------------------------------------
    # [설정 3] 답변하는 AI(LLM) 준비 - Gemini Flash
    # ---------------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro-latest",
        temperature=0, # 0: 사실 기반, 1: 창의적
        google_api_key=api_key
    )

    # ---------------------------------------------------------
    # [설정 4] 체인 연결 (검색 + 답변)
    # ---------------------------------------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True # 근거 문서 확인용
    )

    print("✅ 준비 완료! 질문을 입력하세요. (종료하려면 'exit' 입력)")
    
    # ---------------------------------------------------------
    # [실행] 대화 반복 루프
    # ---------------------------------------------------------
    while True:
        user_input = input("\nUSER > ")
        
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("👋 종료합니다.")
            break
        
        # AI 답변 생성
        result = qa_chain.invoke({"query": user_input})
        print(f"AI > {result['result']}")


if __name__ == "__main__":
    #ingest_docs()
    ask_bot()