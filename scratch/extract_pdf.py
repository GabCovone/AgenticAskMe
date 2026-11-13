import pypdf

reader = pypdf.PdfReader('IR_Exam_projects_BENFENATI2.pdf')
with open('scratch/exam_guidelines_text.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(reader.pages):
        f.write(f"\n\n================ PAGE {i+1} ================\n\n")
        f.write(page.extract_text() or "")
print("Exam guidelines saved to scratch/exam_guidelines_text.txt")

