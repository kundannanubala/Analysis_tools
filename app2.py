import os
import fitz
from google.cloud import aiplatform
from vertexai.vision_models import ImageCaptioningModel, Image
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
TOKEN_LIMIT = int(os.getenv("GEMINI_OUTPUT_TOKEN_LIMIT", 8192))

# AstraDB Configuration
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "pdf_analysis_vector_store"

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

from langchain.schema import Document

def analyze_and_store_pdf(pdf_path, output_folder):
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
    
    # Create the image captioning model
    model = ImageCaptioningModel.from_pretrained("imagetext@001")
    
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
        documents = [{
            "page_content": embedding_input,
            "metadata": {"page_num": page_num}
        }]

        # Convert the dictionaries to Document objects
        documents = [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in documents]

        vector_store.add_documents(documents)

    # Close the document
    doc.close()
    print(f"PDF '{pdf_path}' has been analyzed and embeddings stored.")

# Example usage
if __name__ == "__main__":
    pdf_path = "1_FPAC_NRCS_ClimateAdaptationPlan_2022.pdf"
    output_folder = "ExtractedImages"
    
    analyze_and_store_pdf(pdf_path, output_folder)
