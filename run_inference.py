from load import generate_caption

# 🔍 Change this path to any image you want to test
image_path = "test.jpg"

caption = generate_caption(image_path)
print("Generated Caption:", caption)
