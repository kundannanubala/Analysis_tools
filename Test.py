import fitz
import os

def read_pdf(file_path):
    """
    Read a PDF file from the specified path using PyMuPDF.
    
    Args:
    file_path (str): The full path to the PDF file.
    
    Returns:
    str: The text content of the PDF file.
    """
    try:
        # Open the PDF file
        doc = fitz.open(file_path)
        
        # Initialize an empty string to store the text
        text = ""
        
        # Iterate through all pages and extract text
        for page_num, page in enumerate(doc, 1):
            # Add a page separator
            text += f"\n{'='*50}\n"
            text += f"Page {page_num}\n"
            text += f"{'='*50}\n\n"
            
            # Extract text from the page
            page_text = page.get_text()
            
            # Remove excessive newlines and spaces
            page_text = ' '.join(page_text.split())
            
            # Add the cleaned text to the main text string
            text += page_text + "\n"
        
        # Close the document
        doc.close()
        
        return text
    
    except fitz.FileDataError:
        print(f"The file at {file_path} is not a valid PDF.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    return None

def save_to_file(content, output_file):
    """
    Save the content to a text file.
    
    Args:
    content (str): The text content to save.
    output_file (str): The name of the output file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

# Example usage
if __name__ == "__main__":
    pdf_path = "PP_ISS_REP_961_30APR2009_25MAR2013.pdf"
    output_file = "PDF_Content.txt"
    
    pdf_content = read_pdf(pdf_path)
    
    if pdf_content:
        save_to_file(pdf_content, output_file)
        print(f"PDF content has been saved to {output_file}")
        
        # Print the first 500 characters as a preview
        print("\nPreview of the extracted content:")
        print(pdf_content[:500] + "...")
    else:
        print("Failed to read the PDF file.")
