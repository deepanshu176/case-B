from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_query(persona, job):
    return f"{persona} needs to: {job}"

def score_pages(query, text_pages):
    query_emb = model.encode([query])[0]
    scored = []
    for page in text_pages:
        page_text = page["text"]
        emb = model.encode([page_text])[0]
        score = cosine_similarity([query_emb], [emb])[0][0]
        scored.append({
            "page_number": page["page"],
            "text": page_text,
            "score": score
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True)

def pick_top_sections(scored_pages, file_name, top_n=3):
    return [{
        "document": file_name,
        "page_number": page["page_number"],
        "section_title": f"Section from Page {page['page_number']}",
        "importance_rank": i + 1
    } for i, page in enumerate(scored_pages[:top_n])]

def generate_subsections(scored_pages, file_name, top_n=3):
    highlights = []
    for page in scored_pages[:top_n]:
        lines = page["text"].split("\n")
        for line in lines:
            if len(line.strip()) > 50:
                highlights.append({
                    "document": file_name,
                    "refined_text": line.strip(),
                    "page_number": page["page_number"]
                })
                break
    return highlights
