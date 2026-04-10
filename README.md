# 🔍 EvidentAI: Production-Grade RAG with Automated Quality Gates

EvidentAI is a high-precision Retrieval-Augmented Generation (RAG) system designed for document auditing. Unlike standard chatbots, EvidentAI is engineered for grounded truth, achieving high citation accuracy through a multi-stage retrieval pipeline.

---

## 🚀 Performance Benchmarks

Using a **Golden Dataset of 50 ground-truth questions** (Claude's Constitution Document), the system was optimized from a slow prototype into a production-ready engine.

| Metric              | Initial Prototype        | Optimized System   | Improvement                |
|---------------------|--------------------------|--------------------|----------------------------|
| P99 Latency         | 43.36s                   | 10.26s             | ↓ 76%                      |
| Citation Coverage   | 53.3%                    | 93.7%              | ↑ 40.4%                    |
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
  Top 10 results → reranked → best 5 chunks selected ("Golden 5")

* **Enforced Citation Generation**
  One-shot prompting + Chain-of-Verification ensures every response is grounded with `(Page X)` references.

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
