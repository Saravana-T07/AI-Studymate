import gradio as gr
from transformers import pipeline
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load Granite model
pipe = pipeline("text-generation", model="ibm-granite/granite-3.3-2b-instruct")

# Embedding model for semantic search
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Globals
faiss_index = None
chunks = []

# 1. Process PDFs
def process_pdfs(files):
    global faiss_index, chunks
    chunks = []

    for file in files:
        doc = fitz.open(file.name)
        text = ""
        for page in doc:
            text += page.get_text("text")

        words = text.split()
        step, overlap = 400, 100
        for i in range(0, len(words), step - overlap):
            chunks.append(" ".join(words[i:i+step]))

    # Build FAISS index
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    return f"✅ Processed {len(files)} PDF(s) into {len(chunks)} chunks."

# 2. Retrieve relevant chunks
def retrieve_chunks(query, top_k=3):
    if faiss_index is None:
        return ["⚠️ Please upload PDFs first."]
    query_vec = embedder.encode([query], convert_to_numpy=True)
    _, indices = faiss_index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# 3. Chat function
def chat_fn(history, message):
    retrieved = retrieve_chunks(message)
    context = "\n\n".join(retrieved)

    prompt = f"Answer the question based strictly on the following context:\n{context}\n\nQuestion: {message}\nAnswer:"
    output = pipe(prompt, max_new_tokens=300, temperature=0.5, do_sample=False)
    answer = output[0]["generated_text"]

    # Add references
    answer_with_refs = f"{answer}\n\n📖 **References:**\n- " + "\n- ".join(retrieved)

    history.append((message, answer_with_refs))
    return history, ""

# 4. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📘 StudyMate Chatbox (Granite + PDF Q&A)")

    with gr.Row():
        pdfs = gr.File(label="Upload PDFs", type="file", file_types=[".pdf"], file_types_accept_multiple=True)
    pdf_status = gr.Label()
    pdfs.upload(process_pdfs, pdfs, pdf_status)

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Ask a question about your PDFs")
    clear = gr.Button("Clear Chat")

    state = gr.State([])

    msg.submit(chat_fn, [state, msg], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

demo.launch(share=True)
