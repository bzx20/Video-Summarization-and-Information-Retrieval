# Video-Summarization-and-Information-Retrieval
 This is an AI-based application to answer questions about long videos by segmenting
 them into scenes, indexing for Retrieval-Augmented Generation (RAG), and enabling an
 interactive Q&A interface powered by a Large Language Model (LLM). For example, users
 can quickly grasp a video's main idea by asking for a summary, identify key moments such as
 “When does the presenter compare diffusion models and GANs?” or find specific details like
 “What does the narrator say about the statue’s construction?” — all without needing to watch
 the entire video or scrutinize it frame by frame.

The pipeline, shown in the figure, consists of 4 steps: video summarization, feature extraction,
 indexing, and LLM Q&A interface. It builds on the sample project from Track 1, which
 provides a baseline for video summarization and feature extraction. The sample project
 generates per-clip labels and descriptions, but we found those outputs to be too lengthy to be
 user-friendly. To improve on this, we extend the pipeline with indexing and an LLM-based
 Q&Ainterface. For this additional half of the pipeline, we adapt the methodology of
 GraphRAG to process video content, which is not its typical use case. The official
 GraphRAG examples serve as a good baseline for comparison because they demonstrate its
 effectiveness in more traditional contexts.

 ![image](https://github.com/user-attachments/assets/2a8da945-7023-4e6f-9549-8603d2cf3d9c)

 
