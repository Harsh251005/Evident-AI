# 🔍 EvidentAI: Production-Grade RAG with Automated Quality Gates

EvidentAI is a high-precision Retrieval-Augmented Generation (RAG) system designed for document auditing. Unlike standard chatbots, EvidentAI is engineered for grounded truth, achieving high citation accuracy through a multi-stage retrieval pipeline.

---

## 🚀 Performance Benchmarks

Using a **Golden Dataset of 50 ground-truth questions** (Claude's Constitution Document), the system was optimized from a slow prototype into a production-ready engine.

| Metric              | Initial Prototype        | Optimized System   | Improvement                |
|---------------------|--------------------------|--------------------|----------------------------|
| P99 Latency         | 43.36s                   | 8.57s              | ↓ 80.2%                    |
| P50 Latency         | 33.15s                   | 6.14s              | ↓ 81.5%                    |
| Citation Coverage   | 53%                      | 98%                | ↑ 85%                      |
| Prompting Technique | Zero Shot Prompting      | One Shot Prompting | Improved Citation Accuracy |
| Reranking           | BAAI/bge-reranker-v2-m3  | BGE-Reranker-Base  | Improved Latency           |

---

## 🏗️ System Architecture

EvidentAI follows a **Multi-Stage Retrieval & Refinement Pipeline** to ensure only the most relevant information is passed to the LLM.

### 🔹 Key Components

* **Dynamic Ingestion**
  PDFs are hashed (MD5) to create isolated collections in Qdrant, preventing cross-user data leakage.

* **Hybrid Retrieval**
  Combines:

  * Vector Search (semantic similarity)
  * BM25 (keyword precision)

* **Cross-Encoder Reranking**
  Uses `BAAI/bge-reranker-base` to re-rank top results and select the most relevant chunks.

* **Context Optimization**
  Top 10 results → reranked → best 4 chunks selected ("Golden 4")

* **Enforced Citation Generation**
  One-shot prompting + Chain-of-Verification ensures every response is grounded with `(Page X)` references.

---

## 🎯 Why EvidentAI Was Built

Most beginner RAG systems focus on getting an answer from a document. During experimentation, I noticed that many systems could retrieve relevant information but still failed to provide grounded, verifiable responses.

EvidentAI was built to explore a more production-oriented approach to Retrieval-Augmented Generation by focusing on three core objectives:

1. Retrieval Quality
2. Citation Accuracy
3. Automated Evaluation

The goal was not simply to generate answers, but to ensure that answers could be traced back to supporting evidence from source documents.

This project served as an exploration of evaluation-driven AI development, where retrieval performance, latency, and groundedness were treated as measurable engineering metrics rather than subjective observations.

---

## 🛠️ Tech Stack

* **LLM**: OpenAI GPT-4o-mini
* **Vector Database**: Qdrant (with hashed multi-tenancy)
* **Retriever**: Hybrid (BM25 + Vector Search)
* **Reranker**: BGE Cross-Encoder (HuggingFace)
* **Orchestration**: LangChain
* **Observability & Evaluation**: LangSmith
* **Frontend/UI**: Streamlit
* **Package Manager**: uv

---

## 🤔 Technology Choices

### Why Qdrant?

I chose Qdrant because it is purpose-built for vector search and provides efficient similarity search, metadata filtering, and collection management.

For EvidentAI, I wanted each uploaded document set to be isolated from others. By hashing uploaded PDFs and creating dedicated collections, Qdrant made it straightforward to implement lightweight multi-tenancy while maintaining retrieval performance.

I considered simpler in-memory solutions such as FAISS, but Qdrant was a better fit for a production-oriented architecture because it supports persistence, filtering, and scalable deployment.

---

### Why Hybrid Retrieval?

During testing, I found that vector search and keyword search excelled in different situations.

Vector search performed well for semantic queries:

> "What are the principles of constitutional AI?"

BM25 performed better for highly specific wording:

> "What does the document say about harmlessness?"

Combining both approaches improved retrieval robustness and reduced failure cases where one retrieval strategy alone would miss relevant information.

---

### Why Reranking?

The initial retrieval stage often returned partially relevant chunks.

Instead of increasing context size and sending more information to the LLM, I introduced a cross-encoder reranker to identify the most relevant results before generation.

This significantly improved citation quality while reducing unnecessary context tokens.

---

### Why GPT-4o-mini?

GPT-4o-mini provided a strong balance between:

- Response quality
- Latency
- Cost efficiency

Since the focus of the project was retrieval quality rather than model capability, I prioritized improving retrieval performance before considering larger and more expensive language models.

---

## 🔎 Observability & Evaluation Dashboard

EvidentAI provides full transparency into system behavior using LangSmith.

### 📊 Evaluation Results (Public Dashboard)

You can explore the evaluation dataset results here:

🔗 [https://smith.langchain.com/public/4dbe49ea-ed0d-41cf-8dcc-881bfa25e172/d](https://smith.langchain.com/public/4dbe49ea-ed0d-41cf-8dcc-881bfa25e172/d)

This includes:

* Per-question performance analysis
* Latency per question
* Citation coverage tracking
* Execution traces for debugging

---

## 🛡️ CI/CD Quality Gate

This project includes an automated **AI Quality Gate** to prevent low-quality deployments.

* **Evaluation Suite**: 50 ground-truth questions
* **Threshold**: Minimum 80% citation coverage required
* **Failure Condition**: Build fails if threshold is not met
* **Monitoring**: All runs are logged in LangSmith for debugging and traceability

---

## ⚖️ Design Tradeoffs

### Retrieval Quality vs Latency

Increasing retrieval depth generally improves recall but also increases response latency.

I intentionally limited the final generation stage to the top four reranked chunks ("Golden 4") because it provided a strong balance between context quality and response speed.

---

### Accuracy vs Infrastructure Simplicity

A more advanced architecture could include query rewriting, multi-hop retrieval, and agentic planning.

However, I chose a simpler pipeline that was easier to evaluate, debug, and optimize.

---

### Context Size vs Groundedness

Sending more context to the LLM can increase recall but often introduces irrelevant information.

Instead of maximizing context size, I focused on improving chunk selection quality through reranking and citation enforcement.

---

### Cost vs Performance

Higher-capability language models could potentially improve answer quality.

However, retrieval quality was the larger bottleneck, so I prioritized retrieval optimization before increasing model costs.

---

## 🚧 Challenges Faced

### High Latency

The earliest versions of EvidentAI suffered from response times exceeding 40 seconds.

This was primarily caused by:

- Heavy reranking models
- Inefficient prompting
- Excessive context passed to the LLM

Optimizing retrieval and reducing context size ultimately reduced latency by more than 80%.

---

### Citation Failures

Initial versions frequently generated answers without sufficient grounding.

To address this, I introduced:

- One-shot prompting
- Citation enforcement
- Chain-of-Verification techniques

These changes significantly improved citation coverage.

---

### Retrieval Noise

Many retrieved chunks were only partially relevant.

Introducing reranking improved precision and ensured that only the strongest evidence was passed to the language model.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Harsh251005/Evident-AI.git
cd Evident-AI
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Set Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
QDRANT_URL=http://localhost:6333
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📊 Evaluation & Testing

Run the full evaluation pipeline and quality gate:

```bash
# Step 1: Run evaluations
python -m src.evaluation.run_evals

# Step 2: Run quality gate
$env:LANGSMITH_PROJECT_NAME="your_experiment_name"
python -m src.evaluation.eval_gate
```

---

## 📌 Key Highlights

* Production-grade RAG pipeline
* Hybrid retrieval with reranking
* Automated hallucination control via citation enforcement
* CI/CD integration for AI quality validation
* Strong focus on evaluation-driven development

---

## 🔮 Future Improvements

Given additional development time, I would explore:

- Agentic retrieval workflows
- Query decomposition for complex questions
- Multi-document reasoning
- Adaptive retrieval depth
- Automatic query rewriting
- Domain-specific reranking models
- Human feedback loops for evaluation

I would also like to benchmark EvidentAI against larger RAG systems and evaluate its performance across different document domains beyond constitutional AI.

---

## 📝 What I Would Do Differently

If I were rebuilding EvidentAI today, I would:

- Design the evaluation framework before implementing retrieval logic
- Introduce retrieval diagnostics earlier in development
- Build automated retrieval benchmarks alongside answer-quality benchmarks
- Experiment with late interaction retrieval techniques such as ColBERT
- Add query expansion and decomposition to improve performance on complex questions

The biggest lesson from this project was that improving retrieval quality often has a greater impact on system performance than switching to a larger language model.

---

## 🤝 Contributing

Contributions, ideas, and improvements are welcome. Feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

* OpenAI
* Qdrant
* HuggingFace
* LangChain & LangSmith
