# Corrective RAG (CRAG) — LangGraph Implementation

## Overview

This project implements a **Corrective Retrieval Augmented Generation (CRAG)** pipeline designed to improve answer quality by automatically detecting retrieval failures and correcting them using web search and context refinement.

Traditional RAG systems assume retrieved documents are relevant. This system challenges that assumption by:

* Scoring retrieved chunks
* Detecting incorrect retrieval
* Rewriting queries
* Pulling external knowledge
* Refining context at sentence level
* Generating grounded answers

The entire workflow is orchestrated using **LangGraph**, enabling modular, agent-style reasoning over retrieval quality.

---

## Architecture

### Pipeline Flow

1. **Retrieve**

   * Semantic search from FAISS vector database

2. **Evaluate Retrieved Chunks**

   * LLM assigns relevance score `[0–1]`
   * Classifies retrieval as:

     * `CORRECT`
     * `AMBIGUOUS`
     * `INCORRECT`

3. **Conditional Routing**

   | Verdict   | Action                     |
   | --------- | -------------------------- |
   | CORRECT   | Refine local context       |
   | AMBIGUOUS | Combine local + web        |
   | INCORRECT | Rewrite query → Web search |

4. **Query Rewrite**

   * Converts user question into keyword-focused web query

5. **Web Search**

   * Tavily retrieval fallback

6. **Context Refinement**

   * Sentence decomposition
   * LLM filters only relevant sentences

7. **Answer Generation**

   * Final answer generated strictly from refined context

---

## Graph Structure

```
START
  ↓
Retrieve
  ↓
Evaluate Chunks
  ├── CORRECT → Refine
  └── OTHERWISE → Rewrite Query → Web Search → Refine
  ↓
Generate Answer
  ↓
END
```

---

## Key Features

* Corrective retrieval validation
* Adaptive routing based on retrieval quality
* Sentence-level relevance filtering
* Hybrid knowledge sourcing (local + web)
* Modular LangGraph node design
* Structured output scoring
* Embedding-based semantic search
* Fault-tolerant generation (responds "I don't know")

---

## Tech Stack

* Python
* LangChain
* LangGraph
* FAISS
* HuggingFace Embeddings
* Groq LLM Inference
* Tavily Web Search
* Pydantic

---

## Installation

### 1️⃣ Clone Repository

```
https://github.com/AJINKYA-TEMGIRE/Correct-The-RAG-CRAG-.git
cd Correct-The-RAG-CRAG-
```

### 2️⃣ Create Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

Example requirements:

```
langchain
langgraph
langchain-community
langchain-groq
langchain-huggingface
faiss-cpu
tavily-python
python-dotenv
sentence-transformers
pydantic
```

---

## Environment Variables

Create `.env`

```
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```

---

## Vector Database Setup

Set:

```python
flag = False
```

Run once to build FAISS index from PDFs.

Then revert:

```python
flag = True
```

---

## Running the System

```
python CRAG.py
```

Example query:

```
Batch normalization vs layer normalization
```

Output includes:

* Retrieval verdict
* Reasoning
* Web query used
* Final grounded answer

---

## Project Structure

```
├── Books/
│   ├── book1.pdf
│   ├── book2.pdf
│   └── book3.pdf
│
├── faiss_index_database/
├── main.py
├── requirements.txt
└── README.md
```

---

## Design Philosophy

This implementation focuses on:

* Reliability over blind retrieval
* Explainable routing decisions
* Context precision instead of context volume
* Modular graph extensibility
* Low-latency LLM orchestration

It is intended as a research-grade prototype demonstrating corrective retrieval techniques in agentic pipelines.

---

## Future Improvements

* Parallel document scoring
* Streaming responses
* UI dashboard visualization
* Distributed vector search

---

## Author

Ajinkya Temgire

---

## Acknowledgements

Inspired by recent research directions in:

* Retrieval validation
* Agentic RAG pipelines
* Self-correcting LLM systems
* Knowledge-grounded generation

---

⭐ If you find this useful, consider starring the repo.
