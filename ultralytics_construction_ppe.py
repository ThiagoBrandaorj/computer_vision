import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pathlib import Path

# Download do modelo
ckpt_path = hf_hub_download(
    repo_id="Hansung-Cho/yolov8-ppe-detection",
    filename="best.pt"
)

# Caminhos
image_folder_path = Path(r"C:\Users\kikaj\Documents\IBMEC\7_Periodo\Visao_computacional\AP1\fotos_teste")
output_folder = Path(r"C:\Users\kikaj\Documents\IBMEC\7_Periodo\Visao_computacional\AP1\classified_images")
# Carregar modelo
model = YOLO(ckpt_path)

classes_not_to_use = ['machinery','Mask','NO-Mask','Safety Cone','vehicle']

ids_to_ignore = [id for id, nome in model.names.items()
                 if nome.lower() in [c.lower() for c in classes_not_to_use]]

print(f"Classes a serem excluídas: {classes_not_to_use}")
print(f"IDs correspondentes: {ids_to_ignore}")
print(f"IDs mantidos: {[id for id in model.names.keys() if id not in ids_to_ignore]}")

# Processar imagens
for image_path in image_folder_path.glob("*.jpg"):
    print(f"Processando: {image_path.name}")
    results = model(image_path)
    result = results[0]
    if result.boxes is not None:
        # Filtrar boxes - manter apenas classes NÃO excluídas
        boxes_filtradas = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id not in ids_to_ignore:
                boxes_filtradas.append(box)
        
        print(f"\n{image_path.name}:")
        print(f"  Detecções originais: {len(result.boxes)}")
        print(f"  Detecções após filtro: {len(boxes_filtradas)}")
        
        # Visualizar apenas as boxes desejadas
        img = cv2.imread(str(image_path))
        for box in boxes_filtradas:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            conf = box.conf[0].item()
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} ({conf:.2f})", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
    # Salvar resultado na pasta existente
    cv2.imwrite(str(output_folder / f"classified_{image_path.name}"), img)

print(f"\n✅ Processamento concluído! Resultados salvos em: {output_folder}")