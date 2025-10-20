import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()

# 1. Load your text guideline
file_path = r"D:\HealthCare_ChatBot\fhir-rag-cpg\data\huong-dan-dot-quy-nhe.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents([Document(page_content=text)])

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_text_db")

# 4. Set up retriever + LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 5. Create a QA chain
prompt_template = """You are a medical assistant.  Use the following text from clinical guideline to answer the question. If the answer is not in the text, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 6. Ask questions
queries = [
    "Đột quỵ là gì ?",
    "Triệu chứng lâm sàng của đột quỵ nhẹ bao gồm những gì ?",
    "Chẩn đoán lâm sàng của đột quỵ là gì ?",
]

for query in queries:
    print("=" * 80)
    print(f"Question: {query}")
    result = qa_chain.invoke({"query": query})
    print("\nAnswer:", result["result"])
