import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def load_document(file_path):
    name, extension = os.path.splitext(file_path)
    
    loader_mapping = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader
    }
    
    loader = loader_mapping.get(extension)
    if not loader:
        print('Document format is not supported!')
        return None
    
    return loader(file_path).load()

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

def create_embeddings(chunks, persist_directory='./mychroma_db'):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vector_store

def ask_and_get_answer(vector_store, q, k=6):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': min(k, len(vector_store))})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = chain.invoke({'query': q})
    return response['result'] if 'result' in response else response

def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum(len(enc.encode(page.page_content)) for page in texts)
    return total_tokens, total_tokens / 1000 * 0.0004

def main():
    file_path = input("Enter the path to your file: ")
    data = load_document(file_path)
    
    if data:
        chunk_size = 1000
        chunks = chunk_data(data, chunk_size=chunk_size)
        print(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

        tokens, embedding_cost = calculate_embedding_cost(chunks)
        print(f'Embedding cost: ${embedding_cost:.4f}')

        vector_store = create_embeddings(chunks)
        print('File uploaded, chunked, and embedded successfully.')

        while True:
            q = input("Ask a question about the content of your file (or type 'quit' to exit): ")
            if q.lower() == 'quit':
                break
            answer = ask_and_get_answer(vector_store, q, k=5)
            print("Answer:")
            print(answer.replace('\\n', '\n'))
            print("-" * 100)

if __name__ == "__main__":
    main()
