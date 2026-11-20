# NeuroGraph RAG: Hybrid Graph-Vector Retrieval for Medical Diagnostics

## 📌 About the Project

NeuroGraph RAG is a medical diagnostic system I built as part of my M.Tech (CSE) thesis project. It is a Retrieval-Augmented Generation (RAG) system that combines two ideas:

- **Semantic vector search** to understand the symptoms a user types in natural language.
- **Multi-hop knowledge graph traversal** to connect those symptoms to the most likely diseases.

The main goal of this project is to reduce the "hallucination" problem of LLMs. The model is not allowed to answer from its own memory or from the internet. Every diagnostic answer is based only on a verified clinical dataset.

## 🏗️ How It Works

The system runs in this order:

1. **Vector Database (ChromaDB):** Takes the symptoms written by the user in plain language and finds the closest matching symptoms using semantic similarity.
2. **Knowledge Graph (NetworkX):** Stores the data as `Disease -> HAS_SYMPTOM -> Symptom`. The matched symptoms are mapped on this graph, and candidate diseases are filtered and scored using a rank-based probability method.
3. **Generative AI (Ollama - Llama 3.2):** Gets only the retrieved and verified context and writes a clear, bulleted clinical summary. It does not use any outside knowledge.
4. **Dashboard (Streamlit):** A simple web interface where users can enter symptoms, see the diagnosis, and view the benchmark results.

## 📊 Benchmark Results

I tested the system on **25 real-world clinical test cases** using my evaluation pipeline.

| Metric | Result |
|---|---|
| Top-1 Accuracy | 96.00% |
| Hit@3 | 100.00% |
| Precision (k=3) | 33.33% |
| Clinical Zero-Hallucination Rate | 100.00% |

**Notes:**
- Precision is 33.33% because each test case has only one correct disease, and we return 3 results (k=3). So 33.33% is the maximum possible value here.
- A normal Vector RAG (without the graph step) reaches only **40.00% accuracy** on the same dataset. This shows that graph reasoning makes a big difference.

## 🔐 Login Details

The app has a login page so that only authorised people can use the diagnostic engine. Evaluators can use any one of these accounts:

**Admin**
- Username: `admin`
- Password: `1234`

**User / Doctor**
- Username: `user`
- Password: `1234`

> These are demo credentials only, made for testing and evaluation.

## ⚙️ Prerequisites

Please make sure you have the following before running the project:

- **Python** 3.9 or higher
- **Ollama** installed on your computer
- **Llama 3.2 model**. Download it with:

```bash
ollama pull llama3.2
```

## 🚀 Installation and Setup

Follow these steps to run the project on your computer.

**1. Clone the repository**

```bash
git clone https://github.com/uttam-kumar11/NeuroGraph-RAG.git
cd NeuroGraph-RAG
```

**2. Install the dependencies**

```bash
pip install -r requirements.txt
```

## 💻 Execution Steps

**Step 1: Run the evaluation benchmark (backend check)**

This computes the metrics on the dataset and checks that the whole architecture is working:

```bash
python evaluation.py
```

*This also verifies the database connection and updates the `Data/metrics_results.csv` file.*

**Step 2: Launch the diagnostic dashboard (frontend UI)**

This starts the Streamlit application:

```bash
streamlit run app.py
```

After this, open the link shown in the terminal (usually `http://localhost:8501`) in your browser and log in with the credentials given above.

> Make sure Ollama is running in the background before you start the app, otherwise the summary generation will not work.

## 🛠️ Tech Stack

- **Python**
- **ChromaDB** (vector database)
- **NetworkX** (knowledge graph)
- **Ollama + Llama 3.2** (local LLM)
- **Streamlit** (user interface)

## 🎓 Academic Note

This project was developed for academic and research purposes as part of an M.Tech (CSE) thesis. It is **not** a replacement for a real doctor, and its output should not be used as medical advice.

## 👤 Author

**Uttam Kumar**
M.Tech (CSE)
GitHub: [uttam-kumar11](https://github.com/uttam-kumar11)
,
**Satyam Kumar Sonu**
M.Tech (CSE)
