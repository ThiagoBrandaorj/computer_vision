"""
experimento_objetos_pequenos.py
--------------------------------
Experimento: Detecção de Objetos Pequenos

Metodologia:
  - Analisa a distribuição de tamanho das bounding boxes detectadas
  - Testa diferentes valores de TAMANHO_MINIMO_BOX (threshold)
    para encontrar o equilíbrio entre recall e falsos positivos
  - Simula objetos pequenos fazendo downscale das imagens e
    re-escalando para o tamanho original (simula câmera distante)
  - Registra Precision/Recall aproximados por faixa de tamanho:
      • Pequeno  : área < 32² pixels
      • Médio    : 32² ≤ área < 96² pixels
      • Grande   : área ≥ 96² pixels  (padrão COCO)

  NOTA: Como não temos ground-truth anotado, as métricas aqui são
  APROXIMADAS — baseadas na variação de detecções entre o modelo
  rodando na imagem original (referência) vs. imagens simuladas.
  Para métricas precisas, rotule as imagens no Roboflow e use
  model.val() com o dataset anotado.

Uso:
    python experimento_objetos_pequenos.py --input fotos_teste --output resultados/objetos_pequenos

Dependências:
    pip install ultralytics huggingface_hub opencv-python pandas matplotlib
"""

import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ─── Configurações ────────────────────────────────────────────────────────────

CLASSES_IGNORADAS  = {'machinery', 'mask', 'no-mask', 'safety cone', 'vehicle'}
CONFIANCA_MINIMA   = 0.25

# Fatores de escala para simular objetos distantes (1.0 = original)
FATORES_ESCALA = [1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15]

# Thresholds de tamanho mínimo de box a comparar
THRESHOLDS_BOX = [0, 15, 20, 25, 30, 40, 50, 60]

# Limites de categorias (área em px²)
LIMITE_PEQUENO = 32 * 32   # < 1024 px²
LIMITE_MEDIO   = 96 * 96   # < 9216 px²
# grande >= 9216 px²

# ─── Funções ─────────────────────────────────────────────────────────────────

def carregar_modelo() -> tuple[YOLO, set]:
    ckpt = hf_hub_download(repo_id="Hansung-Cho/yolov8-ppe-detection", filename="best.pt")
    model = YOLO(ckpt)
    ids_ignorados = {
        cid for cid, nome in model.names.items()
        if nome.lower() in CLASSES_IGNORADAS
    }
    return model, ids_ignorados


def categoria_tamanho(area: float) -> str:
    if area < LIMITE_PEQUENO:
        return 'pequeno'
    elif area < LIMITE_MEDIO:
        return 'medio'
    return 'grande'


def inferir_com_threshold(
    model: YOLO,
    img_bgr: np.ndarray,
    ids_ignorados: set,
    min_box: int,
) -> list[dict]:
    """
    Retorna lista de dicts com info de cada detecção válida:
    {class_id, class_name, conf, x1, y1, x2, y2, area, categoria}
    """
    results = model(img_bgr, verbose=False)
    deteccoes = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cid  = int(box.cls[0].item())
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            area = w * h

            if (cid not in ids_ignorados
                    and conf >= CONFIANCA_MINIMA
                    and w >= min_box
                    and h >= min_box):
                deteccoes.append({
                    'class_id'  : cid,
                    'class_name': model.names[cid],
                    'conf'      : round(conf, 4),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'largura'   : w,
                    'altura'    : h,
                    'area'      : area,
                    'categoria' : categoria_tamanho(area),
                })
    return deteccoes


def simular_distancia(img_bgr: np.ndarray, fator: float) -> np.ndarray:
    """
    Faz downscale pelo fator e depois upscale de volta ao tamanho original.
    Simula perda de resolução de objetos distantes.
    """
    h, w = img_bgr.shape[:2]
    pequena = cv2.resize(img_bgr, (max(1, int(w * fator)), max(1, int(h * fator))))
    return cv2.resize(pequena, (w, h), interpolation=cv2.INTER_CUBIC)


def gerar_graficos(df_dist: pd.DataFrame, df_thresh: pd.DataFrame, pasta: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Gráfico 1: detecções por fator de escala ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    media = df_dist.groupby('fator_escala')['n_deteccoes'].mean()
    ax.plot(media.index, media.values, marker='o', color='darkorange', linewidth=2)
    ax.fill_between(media.index, media.values, alpha=0.2, color='darkorange')
    ax.set_xlabel('Fator de Escala (1.0 = original)', fontsize=12)
    ax.set_ylabel('Nº Médio de Detecções Válidas', fontsize=11)
    ax.set_title('Impacto da Distância (Downscale) na Detecção de EPI', fontsize=13, fontweight='bold')
    ax.axvline(x=1.0, linestyle='--', color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(pasta / f"grafico_distancia_{ts}.png", dpi=150)
    plt.close()

    # ── Gráfico 2: detecções por threshold de tamanho mínimo ─────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    media_t = df_thresh.groupby('threshold_box')['n_deteccoes'].mean()
    ax.bar(media_t.index.astype(str), media_t.values, color='steelblue', width=0.6)
    ax.set_xlabel('Threshold Mínimo de Bounding Box (px)', fontsize=12)
    ax.set_ylabel('Nº Médio de Detecções', fontsize=11)
    ax.set_title('Detecções Válidas por Threshold de Tamanho Mínimo', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(pasta / f"grafico_threshold_{ts}.png", dpi=150)
    plt.close()

    # ── Gráfico 3: distribuição de categorias (pizza) ─────────────────────────
    if 'categoria' in df_thresh.columns and not df_thresh.empty:
        df_ref = df_thresh[df_thresh['threshold_box'] == 0]
        cats   = df_ref.groupby('categoria')['n_deteccoes'].mean()
        if not cats.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(cats.values, labels=cats.index, autopct='%1.1f%%',
                   colors=['#e74c3c', '#f39c12', '#2ecc71'])
            ax.set_title('Distribuição de Detecções por Categoria de Tamanho\n(sem threshold)', fontsize=12)
            plt.tight_layout()
            plt.savefig(pasta / f"grafico_categorias_{ts}.png", dpi=150)
            plt.close()

    print(f"📈 Gráficos salvos em {pasta}")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experimento Objetos Pequenos — EPI YOLOv8")
    parser.add_argument('--input',  type=str, default='fotos_teste')
    parser.add_argument('--output', type=str, default='resultados_obj_pequenos')
    args = parser.parse_args()

    pasta_entrada = Path(args.input)
    pasta_saida   = Path(args.output)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    model, ids_ignorados = carregar_modelo()

    extensoes = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    imagens = []
    for ext in extensoes:
        imagens.extend(pasta_entrada.glob(ext))

    if not imagens:
        print(f"⚠️  Nenhuma imagem encontrada em: {pasta_entrada}")
        return

    registros_dist   = []  # variação por fator de escala
    registros_thresh = []  # variação por threshold de box

    total = len(imagens)
    for i, img_path in enumerate(sorted(imagens), 1):
        print(f"[{i}/{total}] {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # ── Experimento A: Simular distância ──────────────────────────────────
        for fator in FATORES_ESCALA:
            img_sim = simular_distancia(img, fator)
            dets    = inferir_com_threshold(model, img_sim, ids_ignorados, min_box=0)
            registros_dist.append({
                'imagem'     : img_path.name,
                'fator_escala': fator,
                'n_deteccoes': len(dets),
            })
            print(f"  escala {fator:.2f} → {len(dets):>3} detecções")

        # ── Experimento B: Comparar thresholds de tamanho mínimo ─────────────
        for thresh in THRESHOLDS_BOX:
            dets = inferir_com_threshold(model, img, ids_ignorados, min_box=thresh)
            cats = {'pequeno': 0, 'medio': 0, 'grande': 0}
            for d in dets:
                cats[d['categoria']] += 1
            registros_thresh.append({
                'imagem'       : img_path.name,
                'threshold_box': thresh,
                'n_deteccoes'  : len(dets),
                **{f'n_{k}': v for k, v in cats.items()},
            })

    df_dist   = pd.DataFrame(registros_dist)
    df_thresh = pd.DataFrame(registros_thresh)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_dist.to_csv(  pasta_saida / f"resultados_distancia_{ts}.csv",  index=False, encoding='utf-8-sig')
    df_thresh.to_csv(pasta_saida / f"resultados_threshold_{ts}.csv",  index=False, encoding='utf-8-sig')
    print(f"\n📊 CSVs salvos em {pasta_saida}")

    # Resumo no terminal
    print("\n─── Impacto do Fator de Escala (média sobre todas as imagens) ───")
    print(df_dist.groupby('fator_escala')['n_deteccoes'].mean().to_string())

    print("\n─── Impacto do Threshold Mínimo de Box ─────────────────────────")
    print(df_thresh.groupby('threshold_box')[['n_deteccoes','n_pequeno','n_medio','n_grande']].mean().to_string())

    gerar_graficos(df_dist, df_thresh, pasta_saida)
    print(f"\n✅ Experimento de objetos pequenos concluído! Resultados em: {pasta_saida}")


if __name__ == '__main__':
    main()
