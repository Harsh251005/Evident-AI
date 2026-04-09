from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from config import settings

# Initialize the models
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=settings.OPENAI_MODEL))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=settings.EMBEDDING_MODEL))

def run_ragas_evaluation(results_list):
    dataset = Dataset.from_list(results_list)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        callbacks=[]
    )

    return result.to_pandas()