import fitz 

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text_pages.append({
            "page": page_num + 1,
            "text": text
        })
    return text_pages
