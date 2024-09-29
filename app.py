import os
import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import aiplatform
from vertexai.vision_models import ImageCaptioningModel, Image
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_astradb import AstraDBVectorStore
from langchain.schema import Document
from pydantic import BaseModel
from typing import Optional
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
TOKEN_LIMIT = int(os.getenv("GEMINI_OUTPUT_TOKEN_LIMIT", 8192))

# AstraDB Configuration
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "pdf_analysis_vector_store"

app = FastAPI()

# Initialize the Vertex AI client
aiplatform.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"))

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

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=0.2, max_output_tokens=8000, top_p=0.95, top_k=40)

class ChatRequest(BaseModel):
    question: str
    source: Optional[str] = None 

@app.post("/analyze-pdf/")
async def analyze_pdf(pdf_file: UploadFile = File(...)):
    """
    Analyze a PDF file: extract text, images, generate image summaries, and store embeddings in AstraDB.
    
    Args:
    pdf_file (UploadFile): The uploaded PDF file.
    
    Returns:
    JSONResponse: A JSON object containing the analysis results.
    """
    # Create a temporary file to store the uploaded PDF
    pdf_path = f"temp_{pdf_file.filename}"
    with open(pdf_path, "wb") as temp_file:
        temp_file.write(await pdf_file.read())
    
    # Create the output folder if it doesn't exist
    output_folder = "ExtractedImages"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Create the image captioning model
    model = ImageCaptioningModel.from_pretrained("imagetext@001")
    
    # Initialize the content dictionary
    content = {"analysis": f"Analysis of {pdf_file.filename}", "pages": []}
    
    # Iterate through each page
    for page_num, page in enumerate(doc, 1):
        page_content = {
            "page_number": page_num,
            "text_content": "",
            "images": []
        }
        
        # Extract text from the page
        page_text = page.get_text()
        page_text = ' '.join(page_text.split())  # Clean up the text
        page_content["text_content"] = page_text
        
        # Extract images from the page and generate captions
        image_list = page.get_images()
        
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
            
            page_content["images"].append({
                "filename": image_filename,
                "summary": summary
            })
        
        content["pages"].append(page_content)
        
        # Generate embedding for this page's content
        embedding_input = f"Text content:\n{page_text}\n\n" + "\n".join([f"Image {i+1}:\n  Caption: {img['summary']}" for i, img in enumerate(page_content['images'])])
        
        # Store the page content and its embedding in AstraDB
        documents = [Document(page_content=embedding_input, metadata={"page_num": page_num, "source": pdf_file.filename})]
        vector_store.add_documents(documents)
    
    # Close the document
    doc.close()
    
    # Remove the temporary PDF file
    os.remove(pdf_path)
    
    return JSONResponse(content=content)


@app.post("/chat_with_pdf")
async def chat_with_pdf_api(chat_request: ChatRequest):
    try:
        # Set up the retriever with optional source filtering
        search_kwargs = {"k": 5}
        if chat_request.source:
            search_kwargs["filter"] = {"source": chat_request.source}
        
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

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
        return {
            "answer": response, 
            "source_filtered": chat_request.source is not None,
            "filter_applied": chat_request.source if chat_request.source else "None"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)