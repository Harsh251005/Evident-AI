from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_texts(doc):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(doc)

    return chunks
