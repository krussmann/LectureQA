from openai import OpenAI
from PyPDF2 import PdfReader
import gensim.downloader as api
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess, compute_sentence_embedding

# Function to extract text from pdf files
def extract_text(pdf_dir, preprocessing):
    pdf_texts = []
    preprocessed_texts = []
    pdf_files = os.listdir(pdf_dir)

    for i, f_name in enumerate(pdf_files):
        file_path = os.path.join(pdf_dir, f_name)
        if 'gitignore' in f_name:
            continue
        try:
            # Open the PDF file
            reader = PdfReader(file_path)
            pages = [page.extract_text().replace("\n", " ") for page in reader.pages]
            # preprocessed = [page for page in pages]
            preprocessed = [preprocessing(page) for page in pages]
            pdf_texts.append(pages)
            preprocessed_texts.append(preprocessed)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return pdf_texts, preprocessed_texts

# Function to find most similar embedding
def similarity_search(question_embedding, pdf_embeddings):
    # Convert question_embedding to a 2D array if it is not already
    question_embedding = np.array(question_embedding).reshape(1, -1)

    # Flatten the pdf_embeddings into a 2D array of dimension (pdf_count*max_page_count, 100)
    flat_embeddings = np.vstack([page for pdf in pdf_embeddings for page in pdf])

    # Compute similarity scores and find most similar embedding
    similarities = cosine_similarity(question_embedding, flat_embeddings).flatten()
    max_idx = np.argmax(similarities)
    max_similarity = similarities[max_idx]

    pdf_idx, page_idx = np.unravel_index(max_idx, pdf_embeddings.shape[:2])
    return pdf_idx, page_idx, max_similarity

def query_llm(question, context):
    api_key = os.environ.get("openai_key")
    if not api_key:
        raise ValueError("API key for OpenAI not found.")
    client = OpenAI(api_key=os.environ["openai_key"])
    prompt = f"Answer the following question based on the attached context. Question: {question} Context: {context}"
    try:
        output = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-3.5-turbo",
        )
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return None

    return output.choices[0].message.content


if __name__ == '__main__':
    # Extract and preprocess pdfs
    pdf_texts, preprocessed = extract_text('Pdfs/', preprocessing=preprocess)
    if not pdf_texts:
        print("No PDFs found or all PDFs failed to process.")
        exit(1)

    # Load embedding model
    glove = api.load("glove-wiki-gigaword-100")

    # Create matrix for pdf embeddings of dimensions (pdf_count, max_page_count, embedding_dim)
    max_page_count = max(len(p) for p in pdf_texts)
    embeddings = np.zeros((len(pdf_texts), max_page_count, 100))

    # Fill matrix with the embeddings
    for i, pdf in enumerate(preprocessed):
        pdf_embedding = np.array([compute_sentence_embedding(slide, glove) for slide in pdf])
        embeddings[i, :len(pdf_embedding), :] = pdf_embedding

    # Query user for question
    question = input("Your question: ").strip()
    question_embedding = compute_sentence_embedding(preprocess(question), glove)

    # Find most similar embedding
    pdf_idx, page_idx, score = similarity_search(question_embedding, embeddings)

    # Print context
    context = pdf_texts[pdf_idx][page_idx]
    pdf_names = os.listdir('Pdfs/')
    if '.gitignore' in pdf_names:
        pdf_names.remove(".gitignore")
    print(f"\nThe answer for this question might be found on page {page_idx + 1} in {pdf_names[pdf_idx]}:\n", context)

    # Print LLM output
    llm_output = query_llm(question, context)
    print("\nLLM answer:")
    print(llm_output)