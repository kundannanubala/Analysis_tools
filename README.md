# Analysis_tools
 A collection of the tools tested and built for analysis

This project currently contains a set of tools for analyzing PDF documents, extracting images, generating summaries, and enabling chat-based interactions with the processed content.

## Features

- PDF text extraction and analysis
- Image extraction from PDFs
- Image captioning using Google's Vertex AI
- Vector storage of document content using AstraDB
- FastAPI-based API for PDF processing and chat interactions
- RAG (Retrieval-Augmented Generation) for answering questions based on PDF content

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/kundannanubala/Analysis_tools.git
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source .venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add the following variables:
     ```
     GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id
     ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
     ASTRA_DB_API_ENDPOINT=your_astra_db_api_endpoint
     MODEL_NAME=gemini-1.5-flash
     GEMINI_OUTPUT_TOKEN_LIMIT=8192
     ```

6. Set up Google Cloud credentials:
   - Place your `credentials.json` file in the root directory
   - Set the environment variable:
     ```

## Usage

### PDF Analysis

To analyze a PDF and store its content:
