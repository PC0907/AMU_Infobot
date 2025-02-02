# AMU Infobot

AMU Infobot is a chatbot application designed to assist students by providing accurate and relevant information about university admissions. The chatbot leverages **Google Gemini AI**, **LangChain**, and **FAISS vector search** to retrieve and process information efficiently.

## 🚀 Features
- **Conversational AI** powered by **Google Gemini**.
- **Retrieval-Augmented Generation (RAG)** for accurate responses.
- **Context-aware conversations** using LangChain memory.
- **PDF document parsing** for extracting relevant admissions details.
- **FAISS vector store** for efficient retrieval of indexed information.

## 🛠️ Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Virtual Environment (optional but recommended)
- Required Python dependencies

### Steps
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/AMU-Infobot.git
   cd AMU-Infobot
2. Create a virtual environment: (optional)
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
4. **Set up environment variables:**
   - Create a `.env` file in the project directory.
   - Add your Google API key:
     ```env
     GOOGLE_API_KEY=your_google_api_key
     ```
5. **Run the application:**
   ```sh
   streamlit run amu_infobot.py



