# Local RAG: The Philosophical Computer

## Project Overview
This project explores the intersection of **technical retrieval systems** and **creative generation**. It is a locally hosted Retrieval-Augmented Generation (RAG) system that ingests course lecture slides (PDFs), builds a semantic knowledge base, and uses a Large Language Model (LLM) to generate "artistic" outputs—specifically, philosophical poetry that uses technical computer science concepts as metaphors for existential struggle.

Unlike standard cloud-based tools, this entire pipeline runs **locally** on your machine, ensuring data privacy and zero cost.

##  Features
- **Local Privacy:** No data is sent to OpenAI or Anthropic; everything runs on your CPU/GPU using Ollama.
- **Document Ingestion:** Automatically scans a directory for PDF slides.
- **Vector Search:** Uses `ChromaDB` and HuggingFace embeddings (`all-MiniLM-L6-v2`) to find relevant technical context.
- **Creative Synthesis:** Summarizes technical content and transforms it into creative writing (e.g., poetry).

## Prerequisites

### 1. Install Ollama
You need **Ollama** installed to run the local LLM.
- Download it from [ollama.com](https://ollama.com).
- Once installed, pull the model used in this project:
  ```bash
  ollama pull gpt-oss:20b
  # OR if that model is unavailable, use llama3:
  ollama pull llama3

### 2. Python Environment
Install the required libraries:
```bash
uv add langchain langchain-community langchain-huggingface langchain-chroma pypdf chromadb
```
