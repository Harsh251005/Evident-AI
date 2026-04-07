from src.evaluation.evaluate import load_dataset, evaluate_system
from main import setup_vector_store, setup_bm25

PDF_PATH = r"D:\Harsh\Code\Resume Projects\EvidentAI\data\sample_pdf\claudes-constitution_webPDF_26-02.02a.pdf"
DATASET_PATH = r"src/evaluation/dataset.json"


def main():
    dataset = load_dataset(DATASET_PATH)

    collection_name, texts, metadata = setup_vector_store(PDF_PATH)
    bm25 = setup_bm25(texts, metadata)

    evaluate_system(dataset, collection_name, bm25)


if __name__ == "__main__":
    main()