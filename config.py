import os; 
from dotenv import load_dotenv; 

#retriev OpenAI api-ky from .env file 
load_dotenv(override=False)

#data paths 
RAW_DATA_DIR = "data\\raw"; 
ARTIFACTS_DIR = "data\\artifacts"; 
DB_PATH = f"{ARTIFACTS_DIR}\\finrag.db"

#retriev from environment variable 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#parameters to process PDF files 
CHUNK_SIZE_TOKENS = 1000; 
OVERLAP_TOKENS = 150; 
TOP_K = 8
