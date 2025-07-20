import fitz  # PyMuPDF
import pandas as pd
from collections import Counter

def extract_features(pdf_path):
    """
    Extracts a feature set for each text block from a given PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return pd.DataFrame()

    font_sizes = Counter()
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        font_sizes[round(s["size"])] += 1
    
    if not font_sizes:
        return pd.DataFrame()

    body_font_size = font_sizes.most_common(1)[0][0]

    features = []
    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                line_text = " ".join([s["text"] for l in b["lines"] for s in l["spans"]]).strip()
                if not line_text:
                    continue

                first_span = b["lines"][0]["spans"][0]
                block_font_size = round(first_span["size"])
                
                feature_dict = {
                    "text": line_text,
                    "font_size": block_font_size,
                    "is_bold": "bold" in first_span["font"].lower(),
                    "is_all_caps": line_text.isupper() and len(line_text) > 1,
                    "y_position": b["bbox"][1] / page_height,
                    "word_count": len(line_text.split()),
                    "relative_size": block_font_size / body_font_size,
                    "page_num": page_num + 1
                }
                features.append(feature_dict)

    return pd.DataFrame(features)