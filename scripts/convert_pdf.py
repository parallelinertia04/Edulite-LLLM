import fitz
import os

input_folder = "C:/Users/Dell/OneDrive/Desktop/EduliteLLM/dataset_raw"
output_folder = "C:/Users/Dell/OneDrive/Desktop/EduliteLLM/dataset_text"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in os.listdir(input_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(input_folder, file)

        print("Processing:", file)

        doc = fitz.open(pdf_path)

        text = ""

        for page in doc:
            text += page.get_text()

        output_file = file.replace(".pdf", ".txt")

        with open(os.path.join(output_folder, output_file), "w", encoding="utf-8") as f:
            f.write(text)

print("PDF extraction completed!")