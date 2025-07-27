import json
from datetime import datetime

def create_final_output(documents, persona, job, sections, subsections):
    result = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job": job,
            "timestamp": datetime.now().isoformat()
        },
        "sections": sections,
        "subsections": subsections
    }
    
    with open("output/output.json", "w") as f:
        json.dump(result, f, indent=4)
