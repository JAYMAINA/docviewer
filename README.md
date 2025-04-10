# PDF Q&A with GPT-4o-mini

A Streamlit application that allows users to upload PDF documents and ask questions about their content using GPT-4o-mini and RAG (Retrieval-Augmented Generation).

## Features

- PDF text extraction
- Text chunking for efficient processing
- Semantic search using FAISS
- Question answering with GPT-4o-mini
- Document summarization

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt

## License

MIT 