# OPTIMIZING CLINICAL DECISIONS: RAG-ENHANCED AI TEXT RETRIEVAL WITH OPENAI AND LANG CHAIN

### Project Description:
Enhancing clinical text retrieval in healthcare, this project leverages advanced methods like OpenAI's GPT-3.5 turbo for language processing and Lang Chain for data indexing. Utilizing the Retrieval-Augmented Generation (RAG) technique and vector databases, it boosts accuracy and efficiency in extracting information from pathology reports crucial for cancer diagnosis and treatment. TruLens evaluates RAG's performance, ensuring precise analysis and refinement, showcasing AI's potential to address healthcare challenges and foster future innovations.

### Objectives:
- Implement the Retrieval-augmented generation (RAG) mechanism.
-  Utilize advanced Natural Language Processing (NLP) techniques.
-  Integrate the Lang Chain framework for connecting language models with data sources.
-  Evaluate the performance of the RAG mechanism using TruLens.

### DATASET:
- Data set: TCGA_Reports.csv
- Content: 9,523 pathology reports providing detailed clinical insights on cancer types, diagnoses, treatments, prognoses, and patient demographics.
- Data set contains two columns:
  - patient_filename: Unique identifier for each patient case.
  - text: Pathology report .Preprocessed data. No missing values


### System Design:

![image](https://github.com/OletiKavya/NLP-RAG_Implementation_clinicalData_TrulensEvaluation/assets/121835613/3e67bef4-12a3-45b2-abc6-9a74cb01e809)


### Methodology:
### RAG IMPLEMENTATION:
#### With Indexing:
- Data Loading: Loaded the 'TCGA_Reports.csv' file using CSV loader from Langchain.
- Data Preprocessing and Chunking:
  - Segmented pathology reports into smaller, manageable chunks.
  - Utilized RecursiveCharacterTextSplitter from Langchain for effective chunking.
- OpenAI Embeddings:
  - Transformed segmented text into embeddings using OpenAI's Text-embedding-3-small.
  - Captured semantic richness and contextual nuances present in the text.
- Pinecone Vector Data Store: Stored preprocessed text and embeddings.
- User Query Embedding and Similarity Search:
  - Embedded user queries using OpenAI embeddings for semantic representation.
  - Conducted similarity search to identify relevant documents or text chunks.
- Generating Responses with GPT 3.5 turbo model:
  - Combined user query with relevant chunk as prompt for generating responses.
  - Utilized GPT 3.5 Turbo model as the language model for improved response quality.
- Streamlit User Interface: Presented results through intuitive Streamlit interface.
- Trulens: Evaluating the RAG model.
#### Without Indexing:
- System relies solely on gpt 3.5 turbo model for response generation
- User query triggers direct invocation of gpt 3.5 model
- Responses synthesized based solely on input query context

### TECHNIQUES:
- LANGCHAIN – Integrating different Data Sources
- NLP  - OPENAI’s gpt 3.5 turbo LLM Model
- RAG –Retrieval Augmented Generation
- TRULENS – Evaluating RAG implementation


### Performance Evaluations: 
Evaluated RAG implementation with Trulens . Observed the results for 20 test cases and found that this implementation has high answer relevance and high context relevance.
![image](https://github.com/OletiKavya/NLP-RAG_Implementation_clinicalData_TrulensEvaluation/assets/121835613/14869631-5119-4b09-8602-58ba777696f8)



### Tools Used
- Programming Language: Python
- Operating System: Windows family
- PineConeand Chroma Data Store
- IDE: Visual Studio Code, Google Colab
- Streamlit

### Libraries: 
- langchain and its associated libraries (langchain-community, langchain-openai, langchain-pinecone)
- Beautiful Soup (bs4)
- Pandas
- NumPy
- Matplotlib and Seaborn
- scikit-learn
- Pinecone


### Attached
- Dataset-TCGA_Reports.csv
- langchain_kavya.ipynb- Containing RAG Implementation [Google Colab]
- graph.py- Streamlit [VSC]
- Trulens_RAG.ipynb- Evaluating RAG [Google colab]


### Conclusion and Future Work :
- Demonstrates RAG's effectiveness in AI conversations.
- High scores for answer and context relevance confirm RAG's accuracy and appropriateness.
- RAG's adaptability allows tailored responses to diverse user needs through dataset adjustments.
- Implement adaptive learning mechanisms for sustained relevance and engagement.
