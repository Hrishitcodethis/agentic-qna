from PyPDF2 import PdfReader
from pytesseract import image_to_string
from PIL import Image
import fitz
import io
from typing import List, Optional
from mcp.server.fastmcp import FastMCP

class PDFExtractor:
    """
    PDFExtractor provides methods to extract text from both normal and scanned PDFs.
    Use extract_content(pdf_path, pages) to get text from a PDF file.
    - pdf_path: Path to the PDF file.
    - pages: Comma-separated string of page numbers (e.g., '1,2,-1').
    Returns extracted text as a string.
    """
    def __init__(self):
        pass

    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Returns True if the PDF is likely scanned (no extractable text), else False.
        """
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if page.extract_text().strip():
                return False
        return True

    def extract_text_from_scanned(self, pdf_path: str, pages: List[int]) -> str:
        """
        Extracts text from scanned PDF pages using OCR.
        """
        doc = fitz.open(pdf_path)
        extracted_text = []
        for page_num in pages:
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = image_to_string(img, lang='chi_sim+eng')
            extracted_text.append(f"Page {page_num + 1}:\n{text}")
        return "\n\n".join(extracted_text)

    def extract_text_from_normal(self, pdf_path: str, pages: List[int]) -> str:
        """
        Extracts text from normal (digitally generated) PDF pages.
        """
        reader = PdfReader(pdf_path)
        extracted_text = []
        for page_num in pages:
            page = reader.pages[page_num]
            extracted_text.append(f"Page {page_num + 1}:\n{page.extract_text()}")
        return "\n\n".join(extracted_text)

    def parse_pages(self, pages_str: Optional[str], total_pages: int) -> List[int]:
        """
        Parses a comma-separated string of page numbers into a list of indices.
        Supports negative indices (e.g., -1 for last page).
        """
        if not pages_str:
            return list(range(total_pages))
        pages = []
        for part in pages_str.split(','):
            if not part.strip():
                continue
            try:
                page_num = int(part.strip())
                if page_num < 0:
                    page_num = total_pages + page_num
                elif page_num > 0:
                    page_num = page_num - 1
                else:
                    raise ValueError("PDF page number cannot be 0")
                if 0 <= page_num < total_pages:
                    pages.append(page_num)
            except ValueError:
                continue
        return sorted(set(pages))

    def extract_content(self, pdf_path: str, pages: Optional[str]) -> str:
        """
        Extracts text from the specified pages of a PDF file.
        Determines if the PDF is scanned or normal and uses the appropriate method.
        Returns extracted text as a string.
        """
        if not pdf_path:
            raise ValueError("PDF path cannot be empty")
        try:
            is_scanned = self.is_scanned_pdf(pdf_path)
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            selected_pages = self.parse_pages(pages, total_pages)
            if is_scanned:
                text = self.extract_text_from_scanned(pdf_path, selected_pages)
            else:
                text = self.extract_text_from_normal(pdf_path, selected_pages)
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract PDF content: {str(e)}")

mcp = FastMCP("pdf_extractor")
extractor = PDFExtractor()

@mcp.tool()
def extract_pdf_contents(pdf_path: str, pages: str = None) -> str:
    return extractor.extract_content(pdf_path, pages)

if __name__ == "__main__":
    mcp.run(transport="stdio")