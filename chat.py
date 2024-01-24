from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import yaml

def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()


load_dotenv()


embedding_function = HuggingFaceEmbeddings(model_name=config["embeddings"]["name"], model_kwargs = {'device': config["embeddings"]["device"]})


db = FAISS.load_local(folder_path=config["faiss_indexstore"]["save_path"], index_name=config["faiss_indexstore"]["index_name"], embeddings = embedding_function)
qa = RetrievalQA.from_llm(llm=ChatOpenAI(temperature=0.9), retriever=db.as_retriever())

while True:
    query = input("Ask me a question: ")
    query = input()
    if query == "quit":
        break
    print(qa.invoke(query))