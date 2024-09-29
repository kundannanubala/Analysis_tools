import fitz
import os

def extract_images(pdf_path, output_folder):
    """
    Extract images from the PDF and save them to the specified folder.
    
    Args:
    pdf_path (str): The path to the PDF file.
    output_folder (str): The folder where images will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
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
    
    # Close the document
    doc.close()

# Example usage
if __name__ == "__main__":
    pdf_path = "1_FPAC_NRCS_ClimateAdaptationPlan_2022.pdf"
    output_folder = "ExtractedImages"
    
    extract_images(pdf_path, output_folder)
    print(f"Images have been extracted to the '{output_folder}' folder.")
