

# 📚 Indic Document Q&A System

This project is an interactive Gradio-based application for processing and querying documents in Indic languages. It allows users to upload, process, and query documents using advanced embedding models and a language model-based chat interface.

## Features

### 1. Document Management
- Upload documents in **PDF** or **TXT** format.
- Extract and process document text into embeddings using:
  - **Bhasantarit** (custom multilingual embedding model).
  - **Sentence Transformer XLM-R**.
- Clear/restart functionality to delete all uploaded documents and indexes to start fresh.

### 2. Chat Interface
- Query uploaded documents to retrieve contextually relevant information.
- Reset the chat interface with a **New Chat** button that clears input, retrieved context, and answer fields.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sandeeppandey456/IndicRAG.git
cd IndicRAG
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Add your Krutrim Cloud API key as an environment variable:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

If you're using a custom API endpoint for Krutrim Cloud, ensure the `base_url` is correctly configured in `document_processor.py`.

### 4. Run the Application
Launch the Gradio interface:
```bash
python app.py
```

The app will be accessible in your browser at `http://localhost:7860`.

---

## Usage Instructions

### Document Management Tab
1. **Upload Document:**
   - Drag and drop a PDF or TXT file into the upload box.
   - The document will be stored and ready for processing.

2. **Select Embedding Model:**
   - Choose between **Bhasantarit** and **Sentence Transformer XLM-R** for embedding generation.

3. **Process Documents:**
   - Click the "Process Documents" button to extract text, generate embeddings, and index them for search.

4. **Clear/Restart:**
   - Use the "Clear/Restart" button to delete all uploaded documents and indexes.

---

### Chat Tab
1. **Ask a Question:**
   - Enter a question in the text box and click "Ask."
   - The app retrieves relevant context from the documents and generates a response using the **Meta-Llama-3.1-70B-Instruct** model.

2. **New Chat:**
   - Use the "New Chat" button to clear all input, retrieved context, and generated answers.

---

## Directory Structure
```
.
├── config.py                # Configuration constants (paths, limits, etc.)
├── document_processor.py    # Handles document processing and embedding generation
├── vector_store.py          # Manages vector indexing and retrieval
├── app.py                   # Main application logic and Gradio UI
├── requirements.txt         # Required Python libraries
├── README.md                # Project documentation
└── data/
    ├── documents/           # Uploaded documents
    └── embeddings/          # Generated embeddings and indexes
```

---

## Requirements

- Python 3.8+
- Gradio
- Transformers
- Faiss
- Pdfplumber
- NumPy

Install all requirements using:
```bash
pip install -r requirements.txt
```

---

## Future Enhancements
1. Support for additional embedding and chat models.
2. OCR based pdf parsing support.
3. Improved document preprocessing for complex PDFs.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

Feel free to update the details in this `README.md` to align with your project specifics.