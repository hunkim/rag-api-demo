import os
from fpdf import FPDF

def convert_txt_to_pdf(txt_path, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    with open(txt_path, 'r', encoding='utf-8') as txt_file:
        for line in txt_file:
            # Ignore encoding errors and replace unsupported characters with '?'
            line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(200, 10, txt=line, align='L')
    
    pdf.output(pdf_path)

def main():
    # Specify the directory path containing the .txt files
    directory = "dataset_en"
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_path = os.path.join(directory, filename)
            pdf_path = os.path.join(directory, filename[:-4] + ".pdf")
            
            convert_txt_to_pdf(txt_path, pdf_path)
            print(f"Converted {filename} to {filename[:-4]}.pdf")

if __name__ == "__main__":
    main()
