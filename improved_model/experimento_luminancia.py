"""
experimento_luminancia.py
--------------------------
Experimento: Variação de Luminância (canal V do espaço HSV)

Metodologia (conforme sugestão do Prof. Rigel):
  - Converte a imagem para HSV (Hue, Saturation, Value)
  - Altera APENAS o canal V (luminância/brilho), mantendo H e S fixos
  - Aplica fator de escala de 0.1 até 1.5 (escuro → superexposto)
  - Para cada nível, roda a inferência YOLOv8 e registra:
      • número de detecções válidas
      • confiança média
      • contagem por classe
  - Gera gráfico (PNG) e CSV com os resultados

Isso permite determinar o threshold mínimo de luminância para
detecção confiável — experimento citado na Seção IV-C do artigo.

Uso:
    python experimento_luminancia.py --input fotos_teste --output resultados/resultados_luminancia

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
TAMANHO_MINIMO_BOX = 30
CONFIANCA_MINIMA   = 0.25

# Fatores de luminância testados (multiplicador do canal V em HSV, clip [0,255])
FATORES_LUMINANCIA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                      1.1, 1.2, 1.3, 1.4, 1.5]

# ─── Funções ─────────────────────────────────────────────────────────────────

def carregar_modelo() -> tuple[YOLO, set]:
    ckpt = hf_hub_download(repo_id="Hansung-Cho/yolov8-ppe-detection", filename="best.pt")
    model = YOLO(ckpt)
    ids_ignorados = {
        cid for cid, nome in model.names.items()
        if nome.lower() in CLASSES_IGNORADAS
    }
    return model, ids_ignorados


def ajustar_luminancia(img_bgr: np.ndarray, fator: float) -> np.ndarray:
    """
    Altera a luminância (canal V) em HSV pelo fator indicado.
    Mantém matiz (H) e saturação (S) inalterados.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * fator, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def inferir(model: YOLO, img_bgr: np.ndarray, ids_ignorados: set) -> tuple[int, float]:
    """
    Roda inferência em array NumPy (já manipulado).
    Retorna (n_deteccoes_validas, confianca_media).
    """
    results = model(img_bgr, verbose=False)
    boxes   = results[0].boxes
    validas = []
    if boxes is not None:
        for box in boxes:
            cid = int(box.cls[0].item())
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (cid not in ids_ignorados
                    and conf >= CONFIANCA_MINIMA
                    and (x2 - x1) >= TAMANHO_MINIMO_BOX
                    and (y2 - y1) >= TAMANHO_MINIMO_BOX):
                validas.append(conf)
    n   = len(validas)
    avg = float(np.mean(validas)) if validas else 0.0
    return n, avg


def salvar_amostra_visual(img_original: np.ndarray, nome: str, pasta: Path):
    """Salva grade com todos os níveis de luminância para inspeção visual."""
    colunas = 5
    linhas  = int(np.ceil(len(FATORES_LUMINANCIA) / colunas))
    h, w    = img_original.shape[:2]
    escala  = 200 / max(h, w)
    th, tw  = int(h * escala), int(w * escala)

    grade = np.zeros((linhas * th, colunas * tw, 3), dtype=np.uint8)
    for idx, fator in enumerate(FATORES_LUMINANCIA):
        r, c = divmod(idx, colunas)
        ajustada = ajustar_luminancia(img_original, fator)
        miniatura = cv2.resize(ajustada, (tw, th))
        cv2.putText(miniatura, f"V×{fator}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        grade[r*th:(r+1)*th, c*tw:(c+1)*tw] = miniatura

    cv2.imwrite(str(pasta / f"grade_luminancia_{nome}"), grade)


def gerar_graficos(df: pd.DataFrame, pasta: Path):
    """Gera gráfico de detecções e confiança média por fator de luminância."""
    df_media = df.groupby('fator_luminancia')[['deteccoes_validas', 'confianca_media']].mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(df_media['fator_luminancia'], df_media['deteccoes_validas'],
            width=0.08, alpha=0.6, color='steelblue', label='Detecções válidas (média)')
    ax2.plot(df_media['fator_luminancia'], df_media['confianca_media'],
             color='tomato', marker='o', linewidth=2, label='Confiança média')

    ax1.set_xlabel('Fator de Luminância (canal V × fator)', fontsize=12)
    ax1.set_ylabel('Nº de Detecções Válidas', color='steelblue', fontsize=11)
    ax2.set_ylabel('Confiança Média', color='tomato', fontsize=11)
    ax1.set_title('Impacto da Luminância na Detecção de EPI', fontsize=13, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.6, label='Luminância original')
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_grafico = pasta / f"grafico_luminancia_{ts}.png"
    plt.savefig(path_grafico, dpi=150)
    plt.close()
    print(f"📈 Gráfico salvo → {path_grafico}")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experimento de Luminância — EPI YOLOv8")
    parser.add_argument('--input',  type=str, default='fotos_teste')
    parser.add_argument('--output', type=str, default='resultados_luminancia')
    args = parser.parse_args()

    pasta_entrada = Path(args.input)
    pasta_saida   = Path(args.output)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    model, ids_ignorados = carregar_modelo()
    print(f"🔬 Fatores de luminância testados: {FATORES_LUMINANCIA}\n")

    extensoes = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    imagens = []
    for ext in extensoes:
        imagens.extend(pasta_entrada.glob(ext))

    if not imagens:
        print(f"⚠️  Nenhuma imagem encontrada em: {pasta_entrada}")
        return

    registros = []
    total_imgs = len(imagens)

    for i, img_path in enumerate(sorted(imagens), 1):
        print(f"[{i}/{total_imgs}] {img_path.name}")
        img_original = cv2.imread(str(img_path))
        if img_original is None:
            print(f"  ⚠️  Não foi possível ler {img_path.name}, pulando.")
            continue

        # Salva grade visual (só para a primeira imagem, para não poluir saída)
        if i == 1:
            salvar_amostra_visual(img_original, img_path.name, pasta_saida)

        for fator in FATORES_LUMINANCIA:
            img_ajustada = ajustar_luminancia(img_original, fator)
            n_det, conf_media = inferir(model, img_ajustada, ids_ignorados)

            print(f"  V×{fator:.1f} → detecções: {n_det:>3} | conf média: {conf_media:.3f}")
            registros.append({
                'imagem'           : img_path.name,
                'fator_luminancia' : fator,
                'deteccoes_validas': n_det,
                'confianca_media'  : round(conf_media, 4),
            })

    df = pd.DataFrame(registros)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = pasta_saida / f"resultados_luminancia_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📊 CSV salvo → {csv_path}")

    # Determinar threshold mínimo (primeiro fator onde detecções > 0)
    df_media = df.groupby('fator_luminancia')['deteccoes_validas'].mean()
    ativos = df_media[df_media > 0]
    if not ativos.empty:
        threshold_min = ativos.index.min()
        print(f"\n🔍 Threshold mínimo de luminância estimado: V×{threshold_min}")
        print("   (primeiro fator onde a média de detecções > 0 sobre o conjunto de teste)")

    gerar_graficos(df, pasta_saida)
    print(f"\n✅ Experimento de luminância concluído! Resultados em: {pasta_saida}")


if __name__ == '__main__':
    main()
