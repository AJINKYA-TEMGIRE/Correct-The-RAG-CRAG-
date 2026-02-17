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
from langchain_community.tools.tavily_search import TavilySearchResults
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

    good_docs : List[Document]
    web_docs : List[Document]
    verdict: str

def retrieve(state : State) -> State:
    q = state["question"]
    result  = retriever.invoke(q)
    return {"docs" : result}


def evaluate_docs(state : State) -> State:
    docs = state["docs"]

    class DocEval(BaseModel):
        score: float

    doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
    )

    evaluate_llm = doc_eval_prompt | llm.with_structured_output(DocEval)

    scores : List[float] = []
    good : List[Document] = []

    for d in docs:
        r = evaluate_llm.invoke({"question" : state["question"] , "chunk" : d.page_content})
        scores.append(r.score)

        if r.score > 0.3:
            good.append(d)

    if any(s > 0.7 for s in scores):
        return {
            "verdict" : "Correct",
            "good_docs" : good,
                }

    if len(scores) > 0 and all(s < 0.3 for s in scores):
        return {
            "good_docs": [],
            "verdict": "Incorrect"
        }
    
    else:
        return {
            "good_docs": good,
            "verdict" : "Ambiguous"
            }

def incorrect(state: State) -> State:

    tavily = TavilySearchResults(max_results=5)

    q = state["question"]  
    results = tavily.invoke({"query": q}) 

    web_docs = []
    for r in results or []:

        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}


def refine(state: State) -> State:
    content = ""
    if state["verdict"] == "Incorrect":
        content = "\n\n".join(d.page_content for d in state["web_docs"]).strip()
    else:
        content = "\n\n".join(d.page_content for d in state["good_docs"]).strip()
    
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


def ambiguous(state : State) -> State:
    return {"answer" : "The documents are ambiguous"}

def check(state: State)-> str:
    if state["verdict"] == "Correct":
        return "refine"
    elif state["verdict"] == "Incorrect":
        return "incorrect"
    else:
        return "ambiguous"


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
g.add_node("evaluate" , evaluate_docs)
g.add_node("incorrect" , incorrect)
g.add_node("ambiguous" , ambiguous)
g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "retrieve")
g.add_edge("retrieve" , "evaluate")
g.add_conditional_edges("evaluate" , check , 
                       {"refine" : "refine" , "incorrect" : "incorrect" , "ambiguous" : "ambiguous"})
g.add_edge("incorrect" , "refine")
g.add_edge("refine" , "generate")
g.add_edge("ambiguous" , END)
g.add_edge("generate", END)

app = g.compile()

answer = app.invoke(
     {
        "question": "Explain about Decision Trees.",
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "strips": [],
        "kept": [],
        "refined_context": "",
        "answer": "",
        "web_docs" : []
    }
)

print(answer["verdict"])
print(answer["answer"])


