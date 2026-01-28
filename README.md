An AI-powered Study Assistant built using Generative AI and Retrieval Augmented Generation (RAG) to help students learn smarter.
The system allows users to upload study material and instantly generate summaries, quizzes, flashcards, and accurate answers directly from their notes.

📌 Problem Statement

Students often struggle to:

Quickly revise large amounts of study material

Create quizzes and revision content from notes

Get reliable answers without searching multiple resources

This project solves the problem by using GenAI + semantic search to provide context aware learning assistance from uploaded documents.




🚀 Key Features

📄 Upload PDF study material

📝 Generate concise, student friendly summaries

❓ Auto-generate Multiple Choice Questions (MCQs)

🧠 Create flashcards in Question Answer format

💬 Ask questions directly from your notes

🔍 Accurate, context-based answers using RAG

🖥️ Interactive and easy to use Streamlit UI




🧠 System Architecture (RAG Pipeline)

User uploads study material (PDF)

Text is extracted and split into chunks

Each chunk is converted into vector embeddings

Embeddings are stored in a FAISS vector database

User queries retrieve the most relevant chunks

A Large Language Model generates responses strictly from retrieved context

This approach significantly reduces hallucinations and improves answer accuracy.



🛠️ Tech Stack

Category	Tools

Programming Language-	Python
UI Framework-	Streamlit
GenAI Framework-	LangChain
LLM-	FLAN-T5 (HuggingFace)
Embeddings-	Sentence Transformers
Vector Database-	FAISS
PDF Processing-	PyPDF


⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-study-assistant.git
cd ai-study-assistant

2️⃣ Create & Activate Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py
