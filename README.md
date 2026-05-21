# KeyBERT-MiniLM-HighPrecision-Extractor
A high-precision Semantic Computing engine built on KeyBERT and Streamlit designed to extract contextually deep keywords and phrases from dense PDF documents. By utilizing sentence-transformer embeddings and Maximal Marginal Relevance (MMR), this pipeline goes beyond simple word frequency to capture the true underlying concepts of unstructured text, effectively minimizing redundancy for advanced document analysis.

## Project Motivation & Objectives

Traditional keyword extraction techniques often rely on statistical word frequencies, such as **TF-IDF** or basic frequency count matrices. While computationally efficient, these methods suffer from a critical flaw: **they lack contextual awareness**. They treat words as isolated data points, completely missing synonyms, conceptual nuances, and the overarching "intent" of the document. For instance, a traditional script cannot natively deduce that "thermal management" and "cooling system" share the same semantic root.

Furthermore, statistical extraction frequently introduces **keyword redundancy**—returning variations of the same core phrase (e.g., *“prediction”*, *“predicting”*, *“predictive”*), which clutters data and skews downstream analysis.

### Core Research Objectives:
* **Contextual Comprehension:** Transition from primitive frequency-counting to deep semantic evaluation by leveraging pre-trained state-of-the-art Transformer models (**BERT**).
* **Information Diversification:** Implement **Maximal Marginal Relevance (MMR)** algorithms to intentionally penalize redundancy, guaranteeing a diverse and conceptually broad keyword output spectrum.
* **Eliminate Hardware Bottlenecks:** Optimize memory allocation and execution times during deep learning inference through local caching decorators (`@st.cache_resource`).

## System Architecture & Workflow
```text
 ┌──────────────────────────┐
 │  Multiple PDF Ingestion  │  ◄── User drops files into Streamlit UI
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │    pdfplumber Parser     │  ◄── Robust text extraction (with Error Handling)
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │  Master Text Unification │  ◄── Content merged and stripped into clean string
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │   KeyBERT AI Engine      │  ◄── Uses cached 'all-MiniLM-L6-v2' weights
 └─────────────┬────────────┘
               │
               ├─► [N-gram Range Generation (1,2)]
               ├─► [Stop-words Removal]
               └─► [MMR Reranking & Diversity Control (0.6)]
               │
               ▼
 ┌──────────────────────────┐
 │  Vector Proximity Table  │  ◄── Computes Cosine Similarity scores
 └─────────────┬────────────┘
               │
               ▼
 ┌──────────────────────────┐
 │  Scrollable UI Viewport  │  ◄── Custom CSS container with instant .CSV export
 └──────────────────────────┘






## Key Features

* **Asynchronous Multi-File Ingestion:** Seamlessly accepts and batch-processes multiple PDF documents simultaneously.
* **In-Memory Transformer Caching:** Utilizes Streamlit's `@st.cache_resource` decorator to store the model weights.
* **Maximal Marginal Relevance (MMR) Pipeline:** Features advanced diversification controls (`diversity=0.6`).
* **Optimized Scrollable Viewport:** Features a custom CSS-styled layout with a fixed-height table container.
* **Precision Floating-Point Formatting:** Automatically formats AI relevance scores to four decimal places.
* **Production-Ready Data Export:** Includes an instant one-click CSV download feature that encodes data cleanly.
* **Fail-Safe Exception Handling:** Built with robust `try-except` blocks around the file-reading layers.

---

## Tech Stack & Dependencies

This research pipeline is engineered using robust, open-source Python libraries optimized for natural language processing.

* **Core Language:** `Python 3.10+`
* **User Interface Framework:** `Streamlit` (For rapid, interactive UI rendering and responsive layout)
* **Semantic AI Engine:** `KeyBERT` (Leveraging the `all-MiniLM-L6-v2` Sentence-Transformer model for embeddings)
* **Data Parsing Toolkit:** `pdfplumber` (Chosen for its extreme precision in character tracking and text extraction)
* **Data Structure Management:** `Pandas` (For managing array matrices and converting keyword vectors)
---

## Installation & Local Setup

Follow these structured steps to replicate this environment and run the keyword extraction pipeline on your local workstation:

### 1. Clone the Repository
Begin by cloning this architecture to your local directory and shifting into the project path:
```bash
git clone [https://github.com/yourusername/KeyBERT-HighPrecision-Extractor.git](https://github.com/yourusername/KeyBERT-HighPrecision-Extractor.git)
cd KeyBERT-HighPrecision-Extractor

### 2. Set Up a Virtual Environment (Recommended)
To keep your global system dependencies clean and isolated, initialize a Python virtual environment:


### For Windows
python -m venv venv
venv\Scripts\activate

### For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Required Dependencies
Install the specific versions of the analytical and machine learning tools required by the pipeline:

pip install streamlit pdfplumber pandas keybert sentence-transformers
### 4. Execute the Application Server
Launch the local Streamlit development server to compile the scripts and launch the web interface in your browser:

streamlit run app.py
Once executed, the terminal will provide a local URL network gateway (usually http://localhost:8501).

##  Algorithmic Tuning Parameters

The analytical precision of this pipeline relies heavily on the structural parameters passed into the KeyBERT core model. Below is the technical breakdown of the configuration used for this research:

*   **`keyphrase_ngram_range=(1, 2)`**: Instructs the model to extract both single-word terms (Unigrams) and dual-word combinations (Bigrams), capturing complex concepts like *"Data Science"* or *"Machine Learning"*.
*   **`stop_words='english'`**: Automatically filters out semantic filler elements (*"the"*, *"is"*, *"and"*) to safeguard vector memory spaces for critical technical terminologies.
*   **`use_mmr=True`**: Activates **Maximal Marginal Relevance**, a text re-ranking strategy that scores candidate words based on text similarity while heavily penalizing informational overlapping.
*   **`diversity=0.6`**: Controls the exact threshold of conceptual novelty. Setting this value to `0.6` establishes a high-performance equilibrium—minimizing redundancy without losing the document's global contextual alignment.
*   **`top_n=30`**: Dictates the definitive array limit, ensuring only the top 30 highest-scoring semantic keyphrases are isolated for user export.

---

##  Future Research Scope

*   **Scalable Vector Databases:** Integrating distributed vector memory structures (like **FAISS** or **Pinecone**) to safely transition the current pipeline from hundreds of pages into thousands of multi-format academic PDFs.
*   **Large Language Model (LLM) Orchestration:** Hooking the extracted semantic metadata directly into Retrieval-Augmented Generation (**RAG**) pipelines via frameworks like **LangChain** or **LlamaIndex** to enable context-aware document conversational indexing.

---

##  License & Contributions

Distributed under the MIT License. Contributions to enhance the performance, processing speed, or structural data capabilities of this repository are highly welcome. Feel free to fork the repository and open a Pull Request.

---

## Connect & Collaborate

This project is part of an ongoing research focus in **Semantic Computing** and **Natural Language Processing**. If you have queries, professional insights, or are interested in collaborative development:

