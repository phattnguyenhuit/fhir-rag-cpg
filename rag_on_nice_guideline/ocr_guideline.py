from paddleocr import PaddleOCR
import os

# Initialize PaddleOCR (optimized config for guideline PDFs)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,  # replaces use_angle_cls
    lang='en'                       # you can change to 'en+ch' or 'en+fr' etc.
)

def ocr_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PaddleOCR."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 🆕 Dùng predict() thay vì ocr()
    result = ocr.predict(file_path)

    extracted_text = []
    for page_idx, page in enumerate(result):
        extracted_text.append(f"\n--- Page {page_idx + 1} ---\n")
        for line in page:
            # ✅ Mỗi line là dictionary: {'transcription': text, 'score': float, ...}
            text = line.get("transcription", "")
            conf = line.get("score", 0)
            if text.strip():
                extracted_text.append(f"{text} (conf: {conf:.2f})")

    return "\n".join(extracted_text)

if __name__ == "__main__":
    pdf_path = r"D:\HealthCare_ChatBot\fhir-rag-cpg\data\stroke-rehabilitation-in-adults-pdf-66143899492549.pdf"
    output_txt = r"D:\HealthCare_ChatBot\fhir-rag-cpg\data\CPG_stroke.txt"

    extracted_text = ocr_pdf(pdf_path)

    # Save text for RAG embedding later
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"📝 Text saved to {output_txt}")