import faiss
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import argparse
from dotenv import load_dotenv
load_dotenv()
parser = argparse.ArgumentParser(description='Ask a question to the Tobig2')
parser.add_argument('question', type=str, help='The question to ask the Tobig2')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(model_name="gpt-4", streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature=0), retriever=store.as_retriever())
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")