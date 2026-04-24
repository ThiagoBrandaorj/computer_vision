from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ckpt_path = hf_hub_download(
    repo_id="Hansung-Cho/yolov8-ppe-detection",
    filename="best.pt"
)

model = YOLO(ckpt_path)

# Método 1: Acessar nomes das classes diretamente
print("Todos os rótulos disponíveis no modelo:")
print("-" * 40)
for id_classe, nome_classe in model.names.items():
    print(f"ID {id_classe}: {nome_classe}")
    
# Método 2: Como dicionário
classes_dict = model.names
print(f"\nTotal de classes: {len(classes_dict)}")