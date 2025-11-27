import os
from dotenv import load_dotenv

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

if __name__ == "__main__":
    ingest_docs()