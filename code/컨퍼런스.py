from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from tqdm import tqdm
import faiss
import pickle

ps = list(Path("best-task-legacy/").glob("**/*.pdf"))     # 컨퍼런스 pdf 파일


data = []
sources = []


### pdf 인코딩
for p in ps:
    if p.exists() and p.is_file():
        loader = PyPDFLoader(str(p))  # p에 있는 pdf 파일을 로드할 수 있는 PyPDFLoader "객체" 생성

        pages = loader.load_and_split()  # PDF 페이지 로드 및 분할
        
        encoded_pages = [str(page).encode('utf-8') for page in pages]  # 페이지를 문자열로 변환 후, UTF-8 인코딩
        
        data.append(encoded_pages)  # "encoded_pages"를 data 리스트에 추가
        
        sources.append(str(p))  # PDF 파일의 "경로"를 문자열로 변환해 sources 리스트에 추가 (원본 파일 경로 추적을 위해)


### 긴 텍스트 나누기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,   # 텍스트 1,500자 단위로 분할
    chunk_overlap=20,   # 각 조각이 20자 겹치게 됨
    length_function=len,   # 텍스트 길이 계산 함수
    is_separator_regex=False,   # 정규 표현식 적용 x
)


### 디코딩
docs = []
metadatas = []
for i, encoded_pages in enumerate(tqdm(data, desc="문서 처리")):
    for encoded_page in encoded_pages:
        
        splits = text_splitter.split_text(encoded_page.decode('utf-8'))  # 인코딩된 페이지를 다시 디코딩하여 문자열 형태로 변환 → split_text에 전달
        
        docs.extend(splits)  # 분할된 텍스트 조각 저장
        
        metadatas.extend([{"source": sources[i]}] * len(splits))  # 각 텍스트 조각에 대한 원본 파일 경로






model_name = "BAAI/bge-large-en-v1.5" # large, base // intfloat/e5-mistral-7b-instruct
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}  # 코사인 유사도를 기반으로 텍스트 간의 유사도를 정확하게 측정하도록 임베딩 정규화
model_norm = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


### FAISS 벡터 저장소 생성 : 문서 텍스트 목록 docs를 받아서 → 임베딩 모델을 사용하여 벡터 생성
store = FAISS.from_texts(docs, model_norm, metadatas=metadatas)

faiss.write_index(store.index, "docs_confernce.index")
store.index = None
with open("faiss_store_confernce.pkl", "wb") as f:
    pickle.dump(store, f)  # store 객체를 직렬화하여 저장 - 나중에 인덱스와 벡터 데이터를 불러올 수 있음