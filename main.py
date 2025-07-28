import os
import glob
import json
import datetime
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

def discover_pdfs(input_dir):
    """Return list of all PDF file paths in the input directory and its subdirectories."""
    pattern = os.path.join(input_dir, "**", "*.pdf")
    pdf_paths = sorted(glob.glob(pattern, recursive=True))
    print("Discovered PDFs:", pdf_paths)  # Debugging output
    return pdf_paths

def load_pdf_text(path):
    """Extract page texts from a PDF. Returns list of dicts: {'page': int, 'text': str}."""
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        pages.append({"page": i + 1, "text": text})
    return pages

def extract_headings_and_sections(pages):
    """
    Naïve section splitter by headings:
    Treat any line in ALL CAPS or ending with ':' as a section heading.
    """
    sections = []
    current_title = f"Page {pages[0]['page']}"
    buffer = []
    for p in pages:
        lines = p["text"].splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            is_heading = (stripped.isupper() and len(stripped) < 100) or stripped.endswith(':')
            if is_heading:
                if buffer:
                    sections.append({"page": p["page"], "title": current_title, "text": "\n".join(buffer)})
                    buffer = []
                current_title = stripped.rstrip(':')
            else:
                buffer.append(stripped)
    if buffer:
        sections.append({"page": pages[-1]["page"], "title": current_title, "text": "\n".join(buffer)})
    return sections

def split_into_subsections(section_text, max_chars=500):
    """Split section text into paragraphs (~subsections) up to max_chars each."""
    paras = section_text.split("\n\n")
    subs = []
    for para in paras:
        text = para.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            subs.append(text)
        else:
            sentences = text.split('. ')
            chunk = ""
            for sent in sentences:
                if len(chunk) + len(sent) + 2 <= max_chars:
                    chunk += (sent + '. ')
                else:
                    subs.append(chunk.strip())
                    chunk = sent + '. '
            if chunk:
                subs.append(chunk.strip())
    return subs

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline PDF relevance extractor (Round 1B)")
    parser.add_argument("--input_dir", required=True, help="Directory containing input PDF files")
    parser.add_argument("--persona", required=True, help="Description of the user persona")
    parser.add_argument("--job", dest="job_to_be_done", required=True, help="Concrete job-to-be-done for the persona")
    parser.add_argument("--output", default="output.json", help="Output JSON filepath")
    args = parser.parse_args()

    pdf_paths = discover_pdfs(args.input_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {args.input_dir}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    pj_text = args.persona + " -- " + args.job_to_be_done
    pj_emb = model.encode(pj_text, convert_to_tensor=True)

    extracted_sections = []
    subsection_analysis = []

    for pdf_path in pdf_paths:
        pages = load_pdf_text(pdf_path)
        sections = extract_headings_and_sections(pages)

        sec_texts = [sec["text"] for sec in sections]
        sec_embs = model.encode(sec_texts, convert_to_tensor=True)

        sims = util.cos_sim(pj_emb, sec_embs)[0]
        ranked_idxs = sims.argsort(descending=True).tolist()

        for rank, idx in enumerate(ranked_idxs, start=1):
            sec = sections[idx]
            extracted_sections.append({
                "document": os.path.basename(pdf_path),
                "page_number": sec["page"],
                "section_title": sec["title"],
                "importance_rank": rank
            })

            subs = split_into_subsections(sec["text"])
            sub_embs = model.encode(subs, convert_to_tensor=True)
            sub_sims = util.cos_sim(pj_emb, sub_embs)[0]
            sub_ranked = sub_sims.argsort(descending=True).tolist()

            for sub_rank, sub_idx in enumerate(sub_ranked, start=1):
                subsection_analysis.append({
                    "document": os.path.basename(pdf_path),
                    "page_number": sec["page"],
                    "refined_text": subs[sub_idx],
                    "importance_rank": sub_rank
                })

    metadata = {
        "input_documents": [os.path.basename(p) for p in pdf_paths],
        "persona": args.persona,
        "job_to_be_done": args.job_to_be_done,
        "processing_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    result = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Output written to {args.output}")

if __name__ == "__main__":
    main()
