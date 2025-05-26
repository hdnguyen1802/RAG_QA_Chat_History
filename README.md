# RAG Chat with PDF

This is a simple chat application that lets you upload PDF documents and ask questions about their content. It uses a RAG (Retrieval Augmented Generation) approach to answer your questions and remembers your chat history.

## Features

* **PDF Upload:** Upload one or more PDF files.
* **Question Answering:** Ask questions based on the content of the uploaded PDFs.
* **Chat History:** The application remembers your previous questions and answers in the current session.

## How to Use

1.  **Get your Gemini API Key:** You will need a Gemini API key to run this application.
2.  **Install Dependencies:**
    Make sure you have Python installed. Then, install the necessary libraries by running:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application:**
    Start the Streamlit application from your terminal:
    ```bash
    streamlit run main.py
    ```
4.  **Open in Browser:**
    Your web browser should automatically open the application. If not, open your browser and go to the local address provided in the terminal.

## Application Interface

* **Sidebar:** On the left, you'll find a sidebar where you can:
    * Enter your **Gemini API key**.
    * Set a **Session ID** (you can leave it as `default_session` if you're just trying it out).
* **Upload PDF Documents:** Use the file uploader to select and upload your PDF files.
* **Chat with PDFs:** Once your PDFs are processed, you can type your questions in the chat box at the bottom of the page.

## Project Structure

* `main.py`: The main application file, handling the Streamlit interface, PDF loading, and chat logic.
* `history_aware_retriever.py`: Contains the function to create a retriever that considers chat history for better context.
* `Youtube_chain.py`: Defines the chain for answering questions based on retrieved documents and chat history.
* `requirements.txt`: Lists all the Python libraries needed to run the application.

## Credits

This application uses the following libraries:
* `langchain` for building the RAG pipeline.
* `streamlit` for the web interface.
* `google-generativeai` for interacting with Gemini models.
* `pypdf` for loading PDF documents.