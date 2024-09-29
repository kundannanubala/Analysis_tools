import os
import base64
import fitz
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageTextModel, Image
from vertexai.vision_models import ImageCaptioningModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
TOKEN_LIMIT = int(os.getenv("GEMINI_OUTPUT_TOKEN_LIMIT", 8192))

def extract_images(pdf_path, output_folder):
    """
    Extract images from the PDF and save them to the specified folder.
    
    Args:
    pdf_path (str): The path to the PDF file.
    output_folder (str): The folder where images will be saved.
    
    Returns:
    list: A list of paths to the extracted images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    extracted_images = []
    
    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get the images on the page
        image_list = page.get_images()
        
        # Iterate through each image
        for img_index, img in enumerate(image_list):
            xref = img[0]  # Get the XREF of the image
            
            # Extract the image
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get the image extension
            image_ext = base_image["ext"]
            
            # Save the image
            image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(output_folder, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            extracted_images.append(image_path)
    
    # Close the document
    doc.close()
    
    return extracted_images

def image_summarize(model: ImageCaptioningModel, image_path: str, prompt: str) -> str:
    """Make image summary"""
    image = Image.load_from_file(image_path)
    captions = model.get_captions(
        image=image,
        number_of_results=1,
        language="en"
    )
    return captions[0] if captions else "No caption generated"

def generate_img_summaries(image_paths: list) -> list[str]:
    """
    Generate summaries for images
    image_paths: List of paths to image files
    """
    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval.
    If it's a table, extract all elements of the table.
    If it's a graph, explain the findings in the graph.
    Do not include any numbers that are not mentioned in the image.
    """

    # Initialize the Vertex AI client
    aiplatform.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"))

    # Create the model
    model = ImageCaptioningModel.from_pretrained("imagetext@001")

    # Apply to images
    for img_path in image_paths:
        image_summaries.append(image_summarize(model, img_path, prompt))

    return image_summaries

if __name__ == "__main__":
    pdf_path = "1_FPAC_NRCS_ClimateAdaptationPlan_2022.pdf"
    output_folder = "ExtractedImages"
    
    # Extract images from PDF
    extracted_image_paths = extract_images(pdf_path, output_folder)
    print(f"Extracted {len(extracted_image_paths)} images from the PDF.")
    
    # Generate summaries for the extracted images
    image_summaries = generate_img_summaries(extracted_image_paths)
    
    # Print summaries
    for i, summary in enumerate(image_summaries):
        print(f"\nSummary for image {i+1}:")
        print(summary)
        print("-" * 50)
