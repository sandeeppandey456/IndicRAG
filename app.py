import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# app.py
import gradio as gr
import numpy as np
from pathlib import Path
from typing import Tuple
import os
from config import DOCUMENTS_DIR, EMBEDDINGS_DIR, MAX_PAGES, MAX_TEXT_LENGTH
from document_processor import DocumentProcessor
from vector_store import VectorStore

class IndicRAG:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def upload_file(self, file: str) -> str:
        try:
            file_path = Path(file)
            if file_path.suffix not in ['.pdf', '.txt']:
                return "Please upload only PDF or TXT files."
            
            dest_path = DOCUMENTS_DIR / file_path.name
            if file_path.suffix == '.pdf':
                text = self.processor.read_pdf(file)
                if not text:
                    return "Could not extract text from PDF."
            else:
                text = file_path.read_text(encoding='utf-8')
                if len(text) > MAX_TEXT_LENGTH:
                    return f"Text file exceeds {MAX_TEXT_LENGTH} characters."
                
            dest_path.write_text(text, encoding='utf-8')
            return f"Successfully uploaded {file_path.name}"
        except Exception as e:
            return f"Error during upload: {str(e)}"

    def process_documents(self, embedding_model: str) -> str:
        try:
            self.processor.embedding_model = embedding_model
            all_text = ""
            for file in DOCUMENTS_DIR.glob("*.*"):
                all_text += file.read_text(encoding='utf-8') + "\n\n"
            
            chunks = self.processor.chunk_text(all_text)
            embeddings = self.processor.create_embeddings(chunks)
            
            # Save processed data
            np.save(EMBEDDINGS_DIR / "chunks.npy", chunks)
            np.save(EMBEDDINGS_DIR / "embeddings.npy", embeddings)
            
            # Create and save index
            index = self.vector_store.create_index(embeddings)
            self.vector_store.save_index(index, str(EMBEDDINGS_DIR / "index.faiss"))
            
            return "Documents processed successfully!"
        except Exception as e:
            return f"Error during processing: {str(e)}"

    def query_documents(self, query: str) -> Tuple[str, str]:
        try:
            index = self.vector_store.load_index(str(EMBEDDINGS_DIR / "index.faiss"))
            chunks = np.load(EMBEDDINGS_DIR / "chunks.npy", allow_pickle=True)
            
            # Get query embedding and search
            query_embedding = self.processor.create_embeddings([query])[0]
            D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
            
            context = "\n\n".join(chunks[i] for i in I[0])
            response = self.processor.get_llama_response(query, context)
            
            return response, context
        except Exception as e:
            return f"Error: {str(e)}", ""

    def clear_documents(self) -> str:
        try:
            # Clear document and embeddings directories
            for file in DOCUMENTS_DIR.glob("*.*"):
                file.unlink()
            for file in EMBEDDINGS_DIR.glob("*.*"):
                file.unlink()
            return "All documents and indexes have been cleared."
        except Exception as e:
            return f"Error clearing documents: {str(e)}"

    def reset_chat(self) -> Tuple[str, str, str]:
        # Reset input, output, and context boxes
        return "", "", ""

    def create_ui(self):
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown("# 📚 Indic Document Q&A System")
            
            with gr.Tab("📄 Document Management"):
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload Document",
                            file_types=[".pdf", ".txt"]
                        )
                        upload_button = gr.Button("Upload", variant="primary")
                        upload_status = gr.Textbox(label="Status", interactive=False)
                        
                        embedding_model = gr.Radio(
                            choices=["Bhasantarit", "Sentence Transformer XLM-R"],
                            value="Bhasantarit",
                            label="Select Embedding Model"
                        )
                        process_button = gr.Button("Process Documents", variant="primary")
                        process_status = gr.Textbox(label="Processing Status", interactive=False)

                        clear_button = gr.Button("Clear/Restart", variant="secondary")
                        clear_status = gr.Textbox(label="Clear Status", interactive=False)
            
            with gr.Tab("💬 Chat"):
                query_input = gr.Textbox(
                    label="Ask a question about your documents",
                    placeholder="Type your question here..."
                )
                with gr.Row():
                    answer_output = gr.Textbox(label="Answer", interactive=False)
                    context_output = gr.Textbox(label="Retrieved Context", interactive=False)
                query_button = gr.Button("Ask", variant="primary")

                reset_button = gr.Button("New Chat", variant="secondary")
            
            # Event handlers
            upload_button.click(
                self.upload_file,
                inputs=[file_upload],
                outputs=[upload_status]
            )
            process_button.click(
                self.process_documents,
                inputs=[embedding_model],
                outputs=[process_status]
            )
            clear_button.click(
                self.clear_documents,
                outputs=[clear_status]
            )
            query_button.click(
                self.query_documents,
                inputs=[query_input],
                outputs=[answer_output, context_output]
            )
            reset_button.click(
                self.reset_chat,
                outputs=[query_input, answer_output, context_output]
            )

        return app

if __name__ == "__main__":
    rag = IndicRAG()
    rag.create_ui().launch()
