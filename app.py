import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from vertexai.vision_models import ImageCaptioningModel, Image
import fitz  # For PDF extraction
import time
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client.models import Document
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Astra DB Configuration
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "pdf_analysis_vector_store"

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize vector store
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    metric="cosine"
)

# Initialize the image captioning model
model = ImageCaptioningModel.from_pretrained("imagetext@001")

# PDF analysis and embedding function
async def analyze_and_store_pdf(pdf_path, output_folder):
    """
    Analyze a PDF file: extract text, images, generate image summaries, and store embeddings in AstraDB.
    
    Args:
    pdf_path (str): The path to the PDF file.
    output_folder (str): The folder where images will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    content_summary = ""  # Store the summary for the API response
    
    # Iterate through each page
    for page_num, page in enumerate(doc, 1):
        # Extract text from the page
        page_text = page.get_text()
        page_text = ' '.join(page_text.split())  # Clean up the text
        
        # Extract images from the page and generate captions
        image_list = page.get_images()
        page_content = f"Text content:\n{page_text}\n\n"
        
        for img_index, img in enumerate(image_list, 1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save the image
            image_filename = f"page{page_num}_img{img_index}.{image_ext}"
            image_path = os.path.join(output_folder, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            # Generate image caption
            image = Image.load_from_file(image_path)
            captions = model.get_captions(image=image, number_of_results=1, language="en")
            summary = captions[0] if captions else "No caption generated"
            page_content += f"Image {img_index}:\n  Caption: {summary}\n\n"
        
        # Combine text and image captions for page-wise embeddings
        embedding_input = page_content.strip()

        # Generate embedding for this page's content
        embedding_vector = embeddings.embed_query(embedding_input)

        # Store the page content and its embedding in AstraDB
        documents = [Document(
            content=embedding_input,
            payload={"embedding": embedding_vector}
        )]
        vector_store.add_documents(
            documents=documents,
            ids=[f"doc_{page_num}"]
        )

        # Append to content summary for response
        content_summary += f"Page {page_num} Summary:\n{page_text[:200]}...\n\n"

    # Close the document
    doc.close()

    return content_summary.strip()


@app.post("/process_pdf")
async def process_pdf_api(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Define the output folder for images
        output_folder = "ExtractedImages"
        
        # Analyze and store the PDF data
        summary = await analyze_and_store_pdf(temp_file_path, output_folder)
        
        # Clean up the temporary file
        os.remove(temp_file_path)

        # Return a summary of the analysis
        return {
            "message": "PDF processed and stored successfully.",
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


class ChatRequest(BaseModel):
    question: str


@app.post("/chat_with_pdf")
async def chat_with_pdf_api(chat_request: ChatRequest):
    try:
        # Set up the retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # Set up the RAG prompt template
        template = """You are an AI assistant tasked with answering questions based on the provided context. 
        Please analyze the context carefully and provide a detailed, accurate answer to the question.

        Context:
        {context}

        Question: {question}

        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        # Set up the RAG chain
        rag_chain = (
            {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Generate the answer
        response = rag_chain.invoke(chat_request.question)
        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
