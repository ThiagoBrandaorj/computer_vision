"""
experimento_oclusao.py
-----------------------
Experimento: Robustez a Oclusões

Metodologia:
  - Simula oclusões sobre as imagens de teste de forma sintética:
      1. Blocos aleatórios (retângulos opacos)  → oclusão aleatória
      2. Faixas horizontais                     → obstáculo tipo grade/cerca
      3. Faixas verticais                       → colunas/pilares
  - Para cada combinação imagem × tipo × intensidade (10%, 20%, … 50%),
    roda a inferência e registra n_detecções e confiança média.
  - Gera gráficos e CSV para análise no artigo (Seção IV).

NOTA: "Intensidade" = fração da área da imagem coberta pela oclusão.

Uso:
    python experimento_oclusao.py --input fotos_teste --output resultados/oclusao

Dependências:
    pip install ultralytics huggingface_hub opencv-python pandas matplotlib
"""

import cv2
import argparse
import random
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
TAMANHO_MINIMO_BOX = 30

# Intensidades de oclusão testadas (fração da área total da imagem)
INTENSIDADES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

# Tipos de oclusão
TIPOS_OCLUSAO = ['blocos', 'faixas_horizontais', 'faixas_verticais']

# Cor de oclusão (preto por padrão)
COR_OCLUSAO = (0, 0, 0)

SEED = 42

# ─── Funções de oclusão ───────────────────────────────────────────────────────

def aplicar_oclusao_blocos(img: np.ndarray, intensidade: float, seed: int = SEED) -> np.ndarray:
    """
    Cobre a imagem com N blocos aleatórios até atingir a intensidade desejada.
    Cada bloco tem tamanho entre 5% e 15% da dimensão menor da imagem.
    """
    if intensidade == 0:
        return img.copy()
    rng = random.Random(seed)
    out = img.copy()
    h, w = img.shape[:2]
    area_alvo = int(intensidade * h * w)
    area_coberta = 0
    tam_base = int(min(h, w) * 0.10)

    while area_coberta < area_alvo:
        bw = rng.randint(tam_base // 2, int(tam_base * 1.5))
        bh = rng.randint(tam_base // 2, int(tam_base * 1.5))
        x  = rng.randint(0, max(0, w - bw))
        y  = rng.randint(0, max(0, h - bh))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), COR_OCLUSAO, -1)
        area_coberta += bw * bh

    return out


def aplicar_oclusao_faixas_horizontais(img: np.ndarray, intensidade: float) -> np.ndarray:
    """Cobre faixas horizontais uniformemente espaçadas."""
    if intensidade == 0:
        return img.copy()
    out = img.copy()
    h, w = img.shape[:2]
    n_faixas  = max(1, int(intensidade * 20))  # 10% → 2 faixas, 50% → 10 faixas
    espessura = max(1, int(h * intensidade / n_faixas))
    passo     = h // n_faixas
    for i in range(n_faixas):
        y = i * passo
        out[y:y + espessura, :] = COR_OCLUSAO
    return out


def aplicar_oclusao_faixas_verticais(img: np.ndarray, intensidade: float) -> np.ndarray:
    """Cobre faixas verticais uniformemente espaçadas."""
    if intensidade == 0:
        return img.copy()
    out = img.copy()
    h, w = img.shape[:2]
    n_faixas  = max(1, int(intensidade * 20))
    espessura = max(1, int(w * intensidade / n_faixas))
    passo     = w // n_faixas
    for i in range(n_faixas):
        x = i * passo
        out[:, x:x + espessura] = COR_OCLUSAO
    return out


FUNCOES_OCLUSAO = {
    'blocos'               : aplicar_oclusao_blocos,
    'faixas_horizontais'   : aplicar_oclusao_faixas_horizontais,
    'faixas_verticais'     : aplicar_oclusao_faixas_verticais,
}


# ─── Inferência ───────────────────────────────────────────────────────────────

def carregar_modelo() -> tuple[YOLO, set]:
    ckpt = hf_hub_download(repo_id="Hansung-Cho/yolov8-ppe-detection", filename="best.pt")
    model = YOLO(ckpt)
    ids_ignorados = {
        cid for cid, nome in model.names.items()
        if nome.lower() in CLASSES_IGNORADAS
    }
    return model, ids_ignorados


def inferir(model: YOLO, img_bgr: np.ndarray, ids_ignorados: set) -> tuple[int, float]:
    results = model(img_bgr, verbose=False)
    confs = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cid  = int(box.cls[0].item())
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (cid not in ids_ignorados
                    and conf >= CONFIANCA_MINIMA
                    and (x2 - x1) >= TAMANHO_MINIMO_BOX
                    and (y2 - y1) >= TAMANHO_MINIMO_BOX):
                confs.append(conf)
    n   = len(confs)
    avg = float(np.mean(confs)) if confs else 0.0
    return n, avg


# ─── Gráficos ─────────────────────────────────────────────────────────────────

def gerar_graficos(df: pd.DataFrame, pasta: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Detecções por intensidade × tipo
    media = df.groupby(['tipo_oclusao', 'intensidade'])['n_deteccoes'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    cores = {'blocos': '#e74c3c', 'faixas_horizontais': '#3498db', 'faixas_verticais': '#2ecc71'}
    for tipo in TIPOS_OCLUSAO:
        sub = media[media['tipo_oclusao'] == tipo]
        ax.plot(sub['intensidade'] * 100, sub['n_deteccoes'],
                marker='o', linewidth=2, label=tipo.replace('_', ' ').title(),
                color=cores.get(tipo))

    ax.set_xlabel('Intensidade de Oclusão (%)', fontsize=12)
    ax.set_ylabel('Nº Médio de Detecções Válidas', fontsize=11)
    ax.set_title('Robustez do Modelo YOLOv8 a Oclusões Sintéticas', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(pasta / f"grafico_oclusao_deteccoes_{ts}.png", dpi=150)
    plt.close()

    # Confiança média
    media_conf = df.groupby(['tipo_oclusao', 'intensidade'])['confianca_media'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    for tipo in TIPOS_OCLUSAO:
        sub = media_conf[media_conf['tipo_oclusao'] == tipo]
        ax.plot(sub['intensidade'] * 100, sub['confianca_media'],
                marker='s', linewidth=2, linestyle='--', label=tipo.replace('_', ' ').title(),
                color=cores.get(tipo))

    ax.set_xlabel('Intensidade de Oclusão (%)', fontsize=12)
    ax.set_ylabel('Confiança Média', fontsize=11)
    ax.set_title('Confiança Média por Tipo e Intensidade de Oclusão', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(pasta / f"grafico_oclusao_confianca_{ts}.png", dpi=150)
    plt.close()

    print(f"📈 Gráficos salvos em {pasta}")


def salvar_amostras_visuais(img: np.ndarray, nome: str, pasta: Path):
    """Salva uma linha de amostras para cada tipo de oclusão com intensidade 0, 20%, 40%."""
    amostras = []
    for tipo in TIPOS_OCLUSAO:
        for intensidade in [0.0, 0.20, 0.40]:
            oculta = FUNCOES_OCLUSAO[tipo](img, intensidade)
            h, w   = oculta.shape[:2]
            escala = 200 / max(h, w)
            mini   = cv2.resize(oculta, (int(w * escala), int(h * escala)))
            cv2.putText(mini, f"{tipo[:5]} {int(intensidade*100)}%", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            amostras.append(mini)

    # Garantir mesma altura
    target_h = min(a.shape[0] for a in amostras)
    amostras = [cv2.resize(a, (int(a.shape[1] * target_h / a.shape[0]), target_h))
                for a in amostras]

    grade = np.concatenate([
        np.concatenate(amostras[0:3], axis=1),
        np.concatenate(amostras[3:6], axis=1),
        np.concatenate(amostras[6:9], axis=1),
    ], axis=0)
    cv2.imwrite(str(pasta / f"amostra_oclusao_{nome}"), grade)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experimento Oclusão — EPI YOLOv8")
    parser.add_argument('--input',  type=str, default='fotos_teste')
    parser.add_argument('--output', type=str, default='resultados_oclusao')
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

    registros = []
    total = len(imagens)

    for i, img_path in enumerate(sorted(imagens), 1):
        print(f"[{i}/{total}] {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if i == 1:
            salvar_amostras_visuais(img, img_path.name, pasta_saida)

        for tipo in TIPOS_OCLUSAO:
            fn = FUNCOES_OCLUSAO[tipo]
            for intensidade in INTENSIDADES:
                img_oculta    = fn(img, intensidade)
                n_det, conf_m = inferir(model, img_oculta, ids_ignorados)
                registros.append({
                    'imagem'        : img_path.name,
                    'tipo_oclusao'  : tipo,
                    'intensidade'   : intensidade,
                    'n_deteccoes'   : n_det,
                    'confianca_media': round(conf_m, 4),
                })
                print(f"  {tipo:<25} {int(intensidade*100):>3}% → det: {n_det:>3} | conf: {conf_m:.3f}")

    df = pd.DataFrame(registros)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = pasta_saida / f"resultados_oclusao_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📊 CSV salvo → {csv_path}")

    print("\n─── Resumo: Detecções médias por tipo e intensidade ────────────")
    pivot = df.groupby(['tipo_oclusao', 'intensidade'])['n_deteccoes'].mean().unstack('intensidade')
    print(pivot.to_string())

    gerar_graficos(df, pasta_saida)
    print(f"\n✅ Experimento de oclusão concluído! Resultados em: {pasta_saida}")


if __name__ == '__main__':
    main()
