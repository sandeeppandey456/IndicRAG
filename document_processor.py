import pdfplumber
from typing import List, Tuple
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import logging
from config import *
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # OpenAI client setup for Krutrim Cloud
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://cloud.olakrutrim.com/v1"
        )
        
        # Transformer model setup
        model_name = "sentence-transformers/stsb-xlm-r-multilingual"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.embedding_model = "Bhasantarit"  # default model

    def read_pdf(self, file) -> str:
        try:
            with pdfplumber.open(file) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages[:MAX_PAGES])
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        chunks = []
        current_chunk = ""
        
        for paragraph in text.split('\n\n'):
            words = paragraph.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= chunk_size:
                    current_chunk += f" {word}" if current_chunk else word
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def create_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        embeddings = []
        for chunk in chunks:
            if self.embedding_model == "Bhasantarit":
                response = self.client.embeddings.create(
                    model="Bhasantarit",
                    input=chunk
                )
                embeddings.append(response.data[0].embedding)
            else:  # Sentence Transformer XLM-R
                inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
        return embeddings

    def get_llama_response(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        response = self.client.chat.completions.create(
            model="Meta-Llama-3.1-70B-Instruct",
            messages=messages
        )
        return response.choices[0].message.content