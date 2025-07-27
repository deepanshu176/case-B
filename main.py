import os
from pdf_utils import extract_text_from_pdf
from ranker import create_query, score_pages, pick_top_sections, generate_subsections
from output_builder import create_final_output

def run_pipeline(pdf_paths, persona, job):
    all_sections = []
    all_subs = []

    for path in pdf_paths:
        file_name = os.path.basename(path)
        pages = extract_text_from_pdf(path)
        query = create_query(persona, job)
        scored = score_pages(query, pages)

        sections = pick_top_sections(scored, file_name)
        subsections = generate_subsections(scored, file_name)

        all_sections.extend(sections)
        all_subs.extend(subsections)

    create_final_output(
        documents=[os.path.basename(p) for p in pdf_paths],
        persona=persona,
        job=job,
        sections=all_sections,
        subsections=all_subs
    )

if __name__ == "__main__":
    pdf_files = [
        "input/paper1.pdf",
    ]
    
    persona = "PhD researcher in Computational Biology"
    job = "Prepare a literature review focusing on methodologies, datasets, and performance benchmarks"
    
    run_pipeline(pdf_files, persona, job)
