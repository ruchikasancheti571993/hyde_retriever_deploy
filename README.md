## Apollo QA with HyDE retriever

The project focuses on answer comparison with and without HyDE retriever

HyDE (Hypothetical Document Embedding) is a zero-shot dense retrieval without the need for relevance labels. This approach involves instructing an LLM to generate a fictional document and then re-encoding it with an unsupervised retriever, like Contriever, to search in its embedding space1.

HyDE has been shown to significantly outperform traditional methods across various tasks and languages, and it does not require any human-labeled relevance judgment

Data- Apollo mission pdfs
