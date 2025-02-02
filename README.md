AMU Infobot

AMU Infobot is a chat-based application designed to assist students with AMU admissions by providing relevant, structured, and accurate responses. It utilizes LangChain, FAISS, and Google Generative AI models to enable intelligent query processing with memory and context retention.

Features

Conversational AI powered by Google Gemini.

Context-aware chat with memory and vector retrieval.

Enhanced document-based knowledge retrieval using FAISS.

Supports PDF document ingestion for domain-specific responses.

Streamlit-based interactive UI for easy access.

Installation

Clone the repository:

git clone https://github.com/yourusername/AMU-Infobot.git
cd AMU-Infobot

Set up a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Set up environment variables:

Create a .env file and add your Google API key:

GOOGLE_API_KEY=your_google_api_key

Ensure the guide to admissions.pdf file is placed correctly.

Usage

Run the application:

streamlit run main.py

Ask admission-related queries in the chat interface.

The bot will respond using its memory and document-based retrieval system.

Technologies Used

LangChain - Conversational AI framework

Google Generative AI (Gemini) - AI-powered responses

FAISS - Vector search for knowledge retrieval

Redis - Chat message history storage

Streamlit - UI framework for chatbot interface

PyPDF2 - PDF processing for knowledge extraction

NumPy - Numerical computations

Contribution

Contributions are welcome! Feel free to fork the repository and create a pull request with improvements.

License

This project is licensed under the MIT License.

Contact

For any queries, contact: [your email or GitHub profile link]

