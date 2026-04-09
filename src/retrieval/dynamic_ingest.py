import os
from src.ingestion.vector_store import generate_collection_name
from src.pipeline.ingestion import ingestion_pipeline


def process_user_upload(uploaded_file):
    os.makedirs("data/indices", exist_ok=True)

    # 1. Save temp file first
    temp_path = f"data/temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Use THE SAME generator as the rest of the app
    collection_name = generate_collection_name(temp_path)

    # 3. Run the pipeline (which now uses the passed name)
    ingestion_pipeline(temp_path, collection_name=collection_name)

    return collection_name