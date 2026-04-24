# 👷 EPI Detector

> Neste projeto usamos como forma de benchmark um modelo pré-treinado de detecção de EPI em canteiros de obras proposto por Hansung-Cho, o projeto no longo prazo visa desenvolver um modelo superior ao protopipo pré treinado usado.

![badge-versão](https://img.shields.io/badge/version-1.0.0-blue)
![badge-licença](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF)
![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-brightgreen)
![screenshot ou demo](classified_images/classified_imagem_web_4.jpg)

## 🦺 Sobre o modelo

O modelo pré treinado em uso está disponível no Hugghing Face na seguinte URL: https://huggingface.co/Hansung-Cho/yolov8-ppe-detection. O modelo é baseado no YOLOv8 e sofreu um fine-tunning pelo autor para detectar as seguintes classes em um canteiro de obras: Hardhat,Mask,NO-Hardhat,NO-Mask,NO-Safety
Vest,Person,Safety Cone,Safety Vest,machinery,vehicle. Porém em nosso contexto usaremos apenas as classes Hardhat,NO-Hardhat,NO-Safety Vest,Person,Safety Vest

## 🧪 Treinamento e métricas do modelo

O modelo foi treinado em conjuntos de dados públicos relacionados a EPI e construção. O resultado de cada inferência do modelo além do rótulo de classe é também a caixa delimitadora do rótulo, determinando onde o modelo está detectando o rótulo destacado. A performance segue na seguinte matriz confusão:
![screenshot ou demo](confusion_matrix.png)

Com base na matriz de confusão do modelo podemos obter as seguintes métricas do modelo:
| Classe | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Hardhat | 57 | 10 | 22 | 85,07% | 72,15% |
| NO-Hardhat | 37 | 8 | 32 | 82,22% | 53,62% |
| NO-Safety Vest | 69 | 16 | 40 | 81,18% | 63,30% |
| Person | 119 | 36 | 44 | 76,77% | 73,01% |
| Safety Vest | 29 | 9 | 8 | 76,32% | 78,38% |

## ⚙️ Uso

Usamos o modelo como forma de teste em algumas imagens, dessas imagens algumas foram obtidas da web e outras foram fotos distantes obtidas de um canteiro de obras com uma câmera semi profissional. As imagens estão na pasta "fotos_teste" e os resultados da classificação do modelo, ou seja, as fotos originais classificadas e com as caixas delimitadoras de objeto estão disponíveis na pasta "classified_images".  

## 🚫 Limitações

Segundo o autor o modelo tem algumas limitações como viés de domínio,oclusão e fotos com baixa resolução.Limitações essas que foram percebidas princiapalmente nas fotos distantes dos objetos.

## 📚 Citação

Cho, Hansung. “YOLOv8 PPE Detection – End-to-End AI System for Safety Monitoring.”
Hugging Face Model Hub, 2025.
