from PIL import Image

image_path = r"C:\Users\kikaj\Documents\IBMEC\7_Periodo\Visao_computacional\AP1\fotos_teste\DSCN7627.JPG"

img = Image.open(image_path)
img_resized = img.resize((1280, 720))
resized_path = image_path.replace(".jpg", "_resized.jpg")
img_resized.save(resized_path)

print(f"Done! Saved to: {resized_path}")