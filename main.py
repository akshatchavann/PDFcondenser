from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from glob import glob
from tqdm import tqdm
import yaml




#load_documents
def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

def load_documents(directory: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config["TextSplitter"]["chunk_size"], 
                                                   chunk_overlap=config["TextSplitter"]["chunk_overlap"])
    documents = []
    for item_path in tqdm(glob(directory + "*.pdf")):
        try:
            loader = PyPDFLoader(item_path)
            loaded_docs = loader.load_and_split(text_splitter=text_splitter)
            if not loaded_docs:
                print(f"No content found in {item_path}")
            documents.extend(loaded_docs)
            print(f"Loaded document from {item_path}")
        except Exception as e:
            print(f"Error loading document from {item_path}: {e}")

    return documents



documents = load_documents("/Users/akshatchavan/Desktop/data/resume")
print("Loaded " + str(len(documents)) + " documents")
 

#load_embeddings
embedding_function = HuggingFaceEmbeddings(model_name=config["embeddings"]["name"], model_kwargs = {'device': config["embeddings"]["device"]})

#load_db
embedding_function = HuggingFaceEmbeddings(model_name=config["embeddings"]["name"], model_kwargs = {'device': config["embeddings"]["device"]})
db = FAISS.from_documents(documents, embedding_function)
db.save_local(config["faiss_indexstore"]["save_path"], config["faiss_indexstore"]["index_name"])


print(db.similarity_search("Akshat's Work Expierence in for Think Throguh"))