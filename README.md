# Legal Document Retrieval using Hybrid RAG for AICUP2024

This repository presents our solution for the [AICUP2024 Legal Document Retrieval Challenge](https://tbrain.trendmicro.com.tw/Competitions/Details/37). The task focuses on retrieving relevant documents for answering legal or financial questions using retrieval-augmented generation (RAG). We employ a hybrid retrieval method combining dense embedding and sparse keyword-based search for optimal document coverage and precision.

## Competition Overview
Competition Name: AICUP 2024 - Legal Document RAG  
AICUP Team ID: TEAM_6459  
Group: 11  
**Leaderboard Rank: 23/487 (Top 4.7%)**

This task emphasizes document relevance and comprehension under a retrieval-augmented generation (RAG) setting.
# Method

## Retrieval Pipeline

We adopt a hybrid retrieval approach:

### Embedding-Based Retrieval
* Model: `intfloat/multilingual-e5-large-instruct` from HuggingFace.
* Tool: `huggingface_hub` and `sentence-transformers` for vector encoding.
* Method: 
  - Query and document chunks are embedded and compared using cosine similarity.
  - Top-K results (K=10) are retrieved based on the highest similarity scores.

### BM25-Based Retrieval
* Library: `rank_bm25.BM25Okapi`
* Method:
  - Documents are tokenized and ranked using classic TF-IDF-style relevance scoring.
  - Top-N results (N=10) are selected.

### Final Document Selection
* Results from both retrieval methods are merged.
* Redundancy is minimized through deduplication heuristics.
* A GPT-4o model (temperature=0.5) is used to rank and select the final document chunk most relevant to the input query.

## Data Preprocessing

### Raw Text Extraction
* PDFs from "financial" and "insurance" domains are parsed using `unstructured.partition.pdf`.
* Extracted texts are concatenated per document.

### Chunking and Cleaning
* Tool: `RecursiveCharacterTextSplitter` (Langchain)
* Strategy:
  - Chunk size: 400 characters
  - Overlap: 200 characters
* Cleaning Steps:
  - Remove special characters
  - Convert Chinese numerals to Arabic using `cn2an`
  - Standardize ROC calendar dates to Gregorian format

### Metadata Management
Each chunk is serialized to JSON and tagged with:
* Chunk ID
* Document source index

## Results

* Retrieval accuracy on internal validation: **0.98**
* Strong performance observed in training data using hybrid retrieval.
* GPT-4o effectively selected final responses based on relevance.

## Reflection and Improvement

### Strengths
* **Comprehensive Preprocessing**: Enhanced consistency and retrieval precision.
* **Ensemble Retrieval Strategy**: Improved coverage via complementary search signals.
* **Chunk-Level Metadata**: Enabled granular control and traceability.

### Limitations
* **Computational Bottlenecks**: Embedding large corpora consumed significant resources.
* **Redundancy**: Overlapping chunks led to occasional output repetition.

### Challenges
* Embedding model selection required empirical testing. Initially selected `multilingual-e5`, later considered `Voyage-3` based on July benchmark reports. Due to license restrictions, reverted to the original model.
* Extracting table data from PDFs was non-trivial. Many tables lacked visible borders, making OCR unreliable. Tried multiple approaches including:
  - Using GPT-4o for structured data extraction
  - Prototyped integration with LLaVA (vision-language model)

## Future Work
* Integrate caching for embedding vectors to speed up testing.
* Refine chunk merging strategy to reduce redundancy.
* Fine-tune embedding models on legal-domain corpora for improved specificity.

## Contact
* Author: Po-Yuan Teng  
* Email: davidteng@g.ncu.edu.tw  
For more details, refer to the code and internal competition report.
