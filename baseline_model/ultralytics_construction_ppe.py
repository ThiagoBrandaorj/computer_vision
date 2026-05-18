import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pathlib import Path
import random

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

# Criar cores aleatórias para cada classe que será mantida
classes_ativas = [id for id in model.names.keys() if id not in ids_to_ignore]
cores_por_classe = {classe_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                    for classe_id in classes_ativas}

# Definir tamanho mínimo da box (em pixels)
TAMANHO_MINIMO_BOX = 30  # Ajuste conforme necessário

# Processar imagens
for image_path in image_folder_path.glob("*.jpg"):
    print(f"Processando: {image_path.name}")
    results = model(image_path)
    result = results[0]
    
    img = cv2.imread(str(image_path))
    altura_img, largura_img = img.shape[:2]
    
    if result.boxes is not None:
        # Filtrar boxes
        boxes_filtradas = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id not in ids_to_ignore:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Calcular tamanho da box
                largura_box = x2 - x1
                altura_box = y2 - y1
                
                # Verificar se atende ao tamanho mínimo
                if largura_box >= TAMANHO_MINIMO_BOX and altura_box >= TAMANHO_MINIMO_BOX:
                    boxes_filtradas.append(box)
        
        print(f"  Detecções originais: {len(result.boxes)}")
        print(f"  Detecções após filtro: {len(boxes_filtradas)}")
        
        # Desenhar boxes com cores por classe
        for box in boxes_filtradas:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            conf = box.conf[0].item()
            
            # Pega a cor específica da classe
            cor = cores_por_classe[class_id]
            
            # Espessura da linha baseada no tamanho da imagem (box maior = linha mais grossa)
            largura_box = x2 - x1
            espessura = max(2, min(5, largura_box // 50))
            
            # Desenhar retângulo
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, espessura)
            
            # Preparar texto
            label = f"{class_name} ({conf:.2f})"
            
            # Tamanho do texto
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Fundo do texto
            cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), cor, -1)
            
            # Texto
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Salvar imagem processada
    cv2.imwrite(str(output_folder / f"classified_{image_path.name}"), img)

print(f"\n✅ Processamento concluído! Resultados salvos em: {output_folder}")
print(f"✅ Tamanho mínimo das boxes: {TAMANHO_MINIMO_BOX} pixels")