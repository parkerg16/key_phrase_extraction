from pdfminer.high_level import extract_text

text = extract_text("data/raw/new_book.pdf")
print(text[:500])