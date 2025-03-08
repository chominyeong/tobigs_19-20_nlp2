## 우수과제 전처리 및 FAISS DB 저장

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from tqdm import tqdm
import faiss
import pickle

model_name = "BAAI/bge-large-en-v1.5" # large, base
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model_norm = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

ps = list(Path("best-task-legacy/").glob("**/*.md"))     # 우수과제 마크다운 파일

data = []
sources = []
for p in ps:
    with open(p, encoding='UTF8') as f:
        data.append(f.read())
    sources.append(p)


text_splitter = RecursiveCharacterTextSplitter(
    separators=["#", "##", "###"],
    chunk_size=1500,  # 텍스트 분할 크기 조절
    chunk_overlap=10)

# 문서 처리 부분
docs = []
metadatas = []
for i, d in enumerate(tqdm(data, desc="문서 처리")):  # tqdm으로 진행률 표시
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

store = FAISS.from_texts(docs, model_norm, metadatas=metadatas)

faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)