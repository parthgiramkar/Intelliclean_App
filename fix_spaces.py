filename = "ul.py"

with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

# Replace ALL non-breaking spaces (U+00A0) with normal spaces
text = text.replace("\u00a0", " ")

with open(filename, "w", encoding="utf-8") as f:
    f.write(text)

print("✅ Cleaned all U+00A0 characters.")
