# RAG Model QA Bot

This project implements a Question Answering (QA) bot using a Retrieval-Augmented Generation (RAG) model. It leverages Pinecone for vector storage, Google's Generative AI for embeddings and chat, and LangChain for various NLP tasks.

## Features

- PDF text extraction
- Text chunking and embedding
- Vector storage using Pinecone
- Question answering using Google's Generative AI
- Streamlit web application for easy interaction

## Prerequisites

- Python 3.10
- Conda (for environment management)
- Pinecone account
- Google AI Platform account (for Gemini API)
- Jupyter Notebook

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a Conda environment:
   ```
   conda create -n ragbot python=3.10
   conda activate ragbot
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following content:
   ```
   GOOGLE_API_KEY="your-gemini-api-key"
   PINECONE_API_KEY="your-pinecone-api-key"
   PDF_FILE="path-to-your-pdf-file"
   ```

5. Set up your Pinecone index:
   - The notebook will create an index named "ragmodeltestbot" with 768 dimensions and cosine similarity metric if it doesn't exist.
   - If you have an existing index with similar configuration, update the `PINECONE_INDEX` variable in the notebook.

## Usage

1. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open the project notebook in the Jupyter interface.

3. Run the cells in the notebook sequentially to process the PDF, create embeddings, and store them in Pinecone.

4. To run the Streamlit app:
   ```
   streamlit run RagApp.py
   ```
   Follow the link provided in the terminal to access the web application.

## Project Structure

- `<your-notebook-name>.ipynb`: Jupyter notebook containing the core functionality for PDF processing, embedding creation, and question answering.
- `RagApp.py`: Streamlit web application for interacting with the QA bot.
- `requirements.txt`: List of Python dependencies.
- `.env`: Environment variables (not tracked in git).

## Important Notes

- Ensure your Pinecone and Google AI Platform accounts are properly set up and have sufficient credits/quota.
- The PDF file path in the notebook is set to "C:/Users/Lenovo/RAG_MODEL_QA_BOT/Documents/WorldWar1.pdf". Update this path according to your file location.
- The Pinecone index is set to use AWS in the us-east-1 region. Modify these settings if needed.
- When running the notebook for the first time, make sure the integrated terminal has the correct directory (i.e., the directory containing the notebook and requirements.txt).

## Troubleshooting

- If you encounter any issues with dependencies, ensure you're using Python 3.10 and have installed all requirements.
- Check that your API keys in the `.env` file are correct and have the necessary permissions.
- If the Pinecone index creation fails, ensure your account has the necessary privileges and your API key is valid.
- If you're having issues running the notebook, make sure Jupyter is installed in your environment: `pip install jupyter`

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[Specify your license here]
