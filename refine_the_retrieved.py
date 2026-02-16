from typing import List, TypedDict
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model = "openai/gpt-oss-120b")
emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

flag = True # currently

if flag == False:
    documents = (
        PyPDFLoader("./Books/book1.pdf").load()
        + PyPDFLoader("./Books/book2.pdf").load()
        + PyPDFLoader("./Books/book3.pdf").load()
    )

    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(documents)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    vectordatabase = FAISS.from_documents(chunks , emb)
    vectordatabase.save_local("faiss_index_database")


database = FAISS.load_local("faiss_index_database" , emb , allow_dangerous_deserialization=True)
retriever = database.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 5}
)


class State(TypedDict):
    question : str
    docs : List[Document]
    strips : List[str]
    kept : List[str]
    refined_context : str
    answer : str

def retrieve(state : State) -> State:
    q = state["question"]
    result  = retriever.invoke(q)
    return {"docs" : result}

def refine(state: State) -> State:

    content = "\n\n".join(d.page_content for d in state["docs"]).strip()
    def decompose_to_sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    strips = decompose_to_sentences(content)

    class KeeporDrop(BaseModel):
        keep : bool

    filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
    )

    chain = filter_prompt | llm.with_structured_output(KeeporDrop)
    kept : List[str] = []
    for s in strips:
        r = chain.invoke({"question" : state["question"] , "sentence" : s})
        if r.keep:
            kept.append(s)

    refined = "\n".join(kept)

    return {
        "kept" : kept,
        "refined_context": refined,    
        "strips" : strips}


def generate(state : State) -> State:
    answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided refined bullets.\n"
            "If the bullets are empty or insufficient, say: 'I don't know based on the provided books.'",
        ),
        ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
    ]
    )
    out = (answer_prompt | llm).invoke({"question": state["question"], "refined_context": state['refined_context']})
    return {"answer" :  out.content}

g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

app = g.compile()

res = app.invoke({
    "question": "Explain the bias–variance tradeoff",
    "docs": [],
    "strips": [],
    "kept_strips": [],
    "refined_context": "",
    "answer": ""
})
print(res["answer"])


