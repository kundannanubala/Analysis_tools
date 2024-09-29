import os
import fitz
from google.cloud import aiplatform
from vertexai.vision_models import ImageCaptioningModel, Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
TOKEN_LIMIT = int(os.getenv("GEMINI_OUTPUT_TOKEN_LIMIT", 8192))

def analyze_pdf(pdf_path, output_folder, output_file):
    """
    Analyze a PDF file: extract text, images, and generate image summaries.
    
    Args:
    pdf_path (str): The path to the PDF file.
    output_folder (str): The folder where images will be saved.
    output_file (str): The name of the output text file.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Initialize the content string
    content = f"Analysis of {pdf_path}\n\n"
    
    # Initialize the Vertex AI client
    aiplatform.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
    
    # Create the image captioning model
    model = ImageCaptioningModel.from_pretrained("imagetext@001")
    
    # Iterate through each page
    for page_num, page in enumerate(doc, 1):
        # Add a page separator
        content += f"\n{'='*50}\n"
        content += f"Page {page_num}\n"
        content += f"{'='*50}\n\n"
        
        # Extract text from the page
        page_text = page.get_text()
        page_text = ' '.join(page_text.split())  # Remove excessive newlines and spaces
        content += f"Text content:\n{page_text}\n\n"
        
        # Extract images from the page
        image_list = page.get_images()
        if image_list:
            content += f"Images on page {page_num}:\n"
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
                
                # Generate image summary
                image = Image.load_from_file(image_path)
                captions = model.get_captions(image=image, number_of_results=1, language="en")
                summary = captions[0] if captions else "No caption generated"
                
                content += f"  Image {img_index}: {image_filename}\n"
                content += f"  Summary: {summary}\n\n"
    
    # Close the document
    doc.close()
    
    # Save the content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"PDF analysis has been saved to {output_file}")
    print(f"Extracted images have been saved to {output_folder}")

# Example usage
if __name__ == "__main__":
    pdf_path = "1_FPAC_NRCS_ClimateAdaptationPlan_2022.pdf"
    output_folder = "ExtractedImages"
    output_file = "PDF_Analysis.txt"
    
    analyze_pdf(pdf_path, output_folder, output_file)
