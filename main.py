from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text

def main():
    pdf_name = input("Enter pdf file name: ")

    document = load_pdf(pdf_name)

    chunks = chunk_text(document)

    print(chunks)

if __name__ == "__main__":
    main()