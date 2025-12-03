import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core. prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER = os.getenv("DOCS_FOLDER")
MODEL_NAME = os.getenv("MODEL_NAME")

def save_markdown(text, filename, title=""):
    """Save text as a nicely formatted markdown file"""
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full path
    filepath = os.path.join(output_dir, filename)
    
    # Build markdown content
    markdown_content = f"# {title}\n\n" if title else ""
    markdown_content += f"{text}\n"
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f. write(markdown_content)
    
    print(f"Saved: {filepath}")

def main():
    print(f"--- Scanning folder: {DOCS_FOLDER} ---")

    loader = PyPDFDirectoryLoader(DOCS_FOLDER)
    docs = loader.load()

    if not docs:
        print("No PDFs found!")
        return

    print(f"Loaded {len(docs)} pages from multiple PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    llm = OllamaLLM(model=MODEL_NAME)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following context to answer the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    chain = (
        RunnableMap({
            "question": lambda x: x["question"],
            "context": lambda x: retriever.invoke(x["question"]),
        })
        | (lambda inputs: {
            "question": inputs["question"],
            "context": "\n\n".join(doc. page_content for doc in inputs["context"])
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run summary
    summary_query = (
        "Synthesize the most important concepts from these slides. "
        "List the top 5 key themes and define them."
    )
    summary = chain.invoke({"question": summary_query})
    print(summary)
    print("\n--- Creative Output ---\n")

    # Run creative output
    creative_query = (
        "Using the following summary of the slides:\n\n"
        f"{summary}\n\n"
        "Write a short philosophical poem about a computer trying to understand "
        "the world, using these key themes and technical terms as metaphors."
    )
    poem = chain.invoke({"question": creative_query})
    print(poem)

    # Save as markdown
    save_markdown(summary, "summary.md", title="Summary of Slides")
    save_markdown(poem, "poem.md", title="Philosophical Poem")
    
if __name__ == "__main__":
    main()