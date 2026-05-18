"""
ppe_inference.py
----------------
Script principal de inferência para detecção de EPI em canteiros de obra.
Utiliza YOLOv8 (Hansung-Cho) com filtro de classes, threshold de bounding box
e exportação automática de métricas para CSV.

Uso:
    python ppe_inference.py --input fotos_teste --output resultados/classified_images

Dependências:
    pip install ultralytics huggingface_hub opencv-python pandas
"""

import cv2
import csv
import argparse
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ─── Configurações globais ────────────────────────────────────────────────────

# Classes que NÃO devem ser detectadas (irrelevantes para segurança do trabalhador)
CLASSES_IGNORADAS = ['machinery', 'Mask', 'NO-Mask', 'Safety Cone', 'vehicle']

# Classes ATIVAS (foco do projeto)
# O modelo de Hansung-Cho detecta: Person, Hardhat, NO-Hardhat, Safety Vest,
# NO-Safety Vest, Mask, NO-Mask, Safety Cone, machinery, vehicle
CLASSES_ATIVAS_ALVO = [
    'Person',
    'Hardhat',
    'NO-Hardhat',
    'Safety Vest',
    'NO-Safety Vest',
]

# Tamanho mínimo da bounding box (largura OU altura em pixels)
TAMANHO_MINIMO_BOX = 30

# Confiança mínima para aceitar uma detecção
CONFIANCA_MINIMA = 0.25

# ─── Funções utilitárias ──────────────────────────────────────────────────────

def carregar_modelo() -> YOLO:
    """Baixa e carrega o modelo YOLOv8 do Hugging Face Hub."""
    print("🔽 Baixando modelo do Hugging Face Hub...")
    ckpt_path = hf_hub_download(
        repo_id="Hansung-Cho/yolov8-ppe-detection",
        filename="best.pt"
    )
    model = YOLO(ckpt_path)
    print(f"✅ Modelo carregado. Classes disponíveis: {list(model.names.values())}\n")
    return model


def configurar_filtros(model: YOLO) -> tuple[set, dict]:
    """
    Retorna:
        ids_ignorados  — set com IDs das classes a descartar
        cores_por_classe — dict {class_id: (B, G, R)}
    """
    ids_ignorados = {
        class_id
        for class_id, nome in model.names.items()
        if nome.lower() in [c.lower() for c in CLASSES_IGNORADAS]
    }

    random.seed(42)  # seed fixa → cores reproduzíveis entre execuções
    cores_por_classe = {
        class_id: (random.randint(40, 230), random.randint(40, 230), random.randint(40, 230))
        for class_id in model.names
        if class_id not in ids_ignorados
    }

    print(f"🚫 Classes ignoradas : {CLASSES_IGNORADAS}")
    print(f"✅ Classes ativas    : {[model.names[i] for i in cores_por_classe]}\n")
    return ids_ignorados, cores_por_classe


def validar_box(box, ids_ignorados: set, confianca_min: float, tamanho_min: int) -> bool:
    """Retorna True se a detecção deve ser mantida."""
    class_id = int(box.cls[0].item())
    if class_id in ids_ignorados:
        return False
    if box.conf[0].item() < confianca_min:
        return False
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    if (x2 - x1) < tamanho_min or (y2 - y1) < tamanho_min:
        return False
    return True


def desenhar_deteccao(img, box, model: YOLO, cores_por_classe: dict):
    """Desenha bounding box + label sobre a imagem."""
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    class_id  = int(box.cls[0].item())
    class_name = model.names[class_id]
    conf       = box.conf[0].item()
    cor        = cores_por_classe.get(class_id, (0, 255, 0))

    largura_box = x2 - x1
    espessura   = max(2, min(5, largura_box // 50))

    cv2.rectangle(img, (x1, y1), (x2, y2), cor, espessura)

    label = f"{class_name} ({conf:.2f})"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), cor, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def processar_imagens(
    model: YOLO,
    ids_ignorados: set,
    cores_por_classe: dict,
    pasta_entrada: Path,
    pasta_saida: Path,
    tamanho_min: int = TAMANHO_MINIMO_BOX,
    confianca_min: float = CONFIANCA_MINIMA,
) -> list[dict]:
    """
    Processa todas as imagens JPG/PNG da pasta de entrada.
    Retorna lista de dicts com métricas por imagem (para exportar CSV).
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)
    extensoes = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    imagens = []
    for ext in extensoes:
        imagens.extend(pasta_entrada.glob(ext))

    if not imagens:
        print(f"⚠️  Nenhuma imagem encontrada em: {pasta_entrada}")
        return []

    print(f"📂 {len(imagens)} imagens encontradas em '{pasta_entrada}'\n")
    registros = []

    for img_path in sorted(imagens):
        results = model(img_path, verbose=False)
        result  = results[0]
        img     = cv2.imread(str(img_path))

        total_brutas  = len(result.boxes) if result.boxes is not None else 0
        boxes_validas = []

        if result.boxes is not None:
            for box in result.boxes:
                if validar_box(box, ids_ignorados, confianca_min, tamanho_min):
                    boxes_validas.append(box)
                    desenhar_deteccao(img, box, model, cores_por_classe)

        # Contagem por classe
        contagem_classe = {nome: 0 for nome in CLASSES_ATIVAS_ALVO}
        for box in boxes_validas:
            nome_classe = model.names[int(box.cls[0].item())]
            if nome_classe in contagem_classe:
                contagem_classe[nome_classe] += 1

        # Salvar imagem classificada
        saida_path = pasta_saida / f"classified_{img_path.name}"
        cv2.imwrite(str(saida_path), img)

        registro = {
            'imagem'         : img_path.name,
            'deteccoes_brutas': total_brutas,
            'deteccoes_validas': len(boxes_validas),
            'descartadas'    : total_brutas - len(boxes_validas),
            **contagem_classe,
        }
        registros.append(registro)

        print(f"  {img_path.name:<35} | bruto: {total_brutas:>3} | válido: {len(boxes_validas):>3} | "
              + " | ".join(f"{k}: {v}" for k, v in contagem_classe.items()))

    return registros


def exportar_csv(registros: list[dict], pasta_saida: Path):
    """Salva métricas em CSV."""
    if not registros:
        return
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = pasta_saida / f"metricas_inferencia_{ts}.csv"
    df = pd.DataFrame(registros)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📊 Métricas exportadas → {csv_path}")

    # Resumo no terminal
    print("\n─── Resumo por classe ───────────────────────────────")
    for col in CLASSES_ATIVAS_ALVO:
        if col in df.columns:
            print(f"  {col:<20}: total={df[col].sum():>4} | média/img={df[col].mean():.2f}")
    print("─────────────────────────────────────────────────────")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inferência de EPI com YOLOv8")
    parser.add_argument('--input',  type=str, default='fotos_teste',
                        help='Pasta com as imagens de entrada')
    parser.add_argument('--output', type=str, default='classified_images',
                        help='Pasta de saída para imagens classificadas')
    parser.add_argument('--min-box',  type=int,   default=TAMANHO_MINIMO_BOX,
                        help='Tamanho mínimo da bounding box em pixels')
    parser.add_argument('--min-conf', type=float, default=CONFIANCA_MINIMA,
                        help='Confiança mínima para aceitar detecção (0–1)')
    args = parser.parse_args()

    pasta_entrada = Path(args.input)
    pasta_saida   = Path(args.output)

    model = carregar_modelo()
    ids_ignorados, cores = configurar_filtros(model)

    registros = processar_imagens(
        model, ids_ignorados, cores,
        pasta_entrada, pasta_saida,
        tamanho_min=args.min_box,
        confianca_min=args.min_conf,
    )

    exportar_csv(registros, pasta_saida)
    print(f"\n✅ Concluído! Resultados em: {pasta_saida}")


if __name__ == '__main__':
    main()
