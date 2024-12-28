# Document QA System

This project provides a Document QA (Question-Answering) system using OpenAI's language models and embeddings. The system enables users to upload a document (PDF, DOCX, or TXT), process it into chunks, generate embeddings, and ask questions about its content. The system uses LangChain and Chroma for embedding storage and retrieval.

## Features
- **Document Upload**: Supports `.pdf`, `.docx`, and `.txt` file formats.
- **Chunking**: Splits documents into manageable chunks for embedding.
- **Embeddings**: Uses OpenAI embeddings for semantic understanding.
- **Question Answering**: Allows users to ask questions and get contextually relevant answers.
- **Cost Estimation**: Calculates the cost of embedding generation based on the number of tokens.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.8 or higher
- Required Python libraries (see [Installation](#installation))
- OpenAI API key

## Installation

1. Clone the repository or copy the script to your local environment.
2. Install the required Python libraries:
   ```bash
   pip install langchain langchain-openai chromadb tiktoken python-dotenv
   ```
3. Create a `.env` file in the project directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Run the script:
   ```bash
   python script_name.py
   ```

2. Follow the prompts to:
   - Upload a document by providing its file path.
   - The script will load, chunk, and generate embeddings for the document.
   - Ask questions about the document's content.

3. To exit the question-answering loop, type `quit`.

## Code Overview

### Functions

- **`load_document(file_path)`**:
  - Loads the document based on its file extension.
  - Supported formats: `.pdf`, `.docx`, `.txt`.

- **`chunk_data(data, chunk_size, chunk_overlap)`**:
  - Splits the document into smaller chunks for embedding.

- **`create_embeddings(chunks, persist_directory)`**:
  - Generates and stores embeddings in a Chroma vector store.

- **`ask_and_get_answer(vector_store, q, k)`**:
  - Processes a user query and retrieves the most relevant answer using OpenAI models.

- **`calculate_embedding_cost(texts)`**:
  - Calculates the cost of generating embeddings based on token usage.

### Main Workflow

1. Load a document.
2. Chunk the document.
3. Generate embeddings and store them.
4. Enter a loop to answer user queries about the document.

## Limitations

- The script only supports `.pdf`, `.docx`, and `.txt` formats.
- Requires an active OpenAI API key for embeddings and question answering.
- Embedding costs depend on the document size and token usage.

## Future Improvements
- Add support for additional file formats.
- Optimize chunking and embedding processes for large documents.
- Provide a web-based or GUI interface for better user interaction.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [OpenAI](https://openai.com/) for their APIs.
- [LangChain](https://github.com/hwchase17/langchain) for the framework.
- [Chroma](https://github.com/chroma-core/chroma) for vector database integration.
