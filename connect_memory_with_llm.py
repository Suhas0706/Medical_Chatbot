import os
from dotenv import load_dotenv

# Correct modern LangChain imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
# Make sure this matches your .env file variable name
HF_TOKEN = os.environ.get("HF_TOKEN") 
HUGGINGFACE_REPO_ID = "deepseek-ai/DeepSeek-V3"


def load_llm():
    # Fixed: Passing the actual HF_TOKEN variable instead of the string "HF_TOKEN"
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN, 
        max_new_tokens=512,
        temperature=0.5,
    )
    return ChatHuggingFace(llm=llm)


prompt = ChatPromptTemplate.from_template(
    """
Use the following context to answer the user's question.

If you don't know the answer, simply say you don't know.

Context:
{context}

Question:
{input}
"""
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Ensure the local directory "vectorstore/db_faiss" actually exists before running
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever(search_kwargs={"k": 3})

question_answer_chain = create_stuff_documents_chain(
    load_llm(),
    prompt,
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain,
)

if __name__ == "__main__":
    user_query = input("Write Query Here: ")
    response = rag_chain.invoke({"input": user_query})
    
    # Fixed: create_retrieval_chain outputs "answer" and "context"
    print("\nRESULT:\n", response.get("answer"))
    print("\nSOURCE DOCUMENTS:\n", response.get("context"))