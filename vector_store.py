import faiss
import numpy as np
from typing import List

class VectorStore:
    @staticmethod
    def create_index(embeddings: List[np.ndarray]) -> faiss.Index:
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index

    @staticmethod
    def save_index(index: faiss.Index, path: str):
        faiss.write_index(index, path)

    @staticmethod
    def load_index(path: str) -> faiss.Index:
        return faiss.read_index(path)