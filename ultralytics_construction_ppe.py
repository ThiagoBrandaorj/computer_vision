from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ckpt_path = hf_hub_download(
    repo_id="Hansung-Cho/yolov8-ppe-detection",
    filename="best.pt"
)
image_path = r"C:\Users\kikaj\Documents\IBMEC\7_Periodo\Visao_computacional\AP1\fotos_teste\DSCN7622.JPG"
model = YOLO(ckpt_path)
results = model(image_path)
results[0].show()