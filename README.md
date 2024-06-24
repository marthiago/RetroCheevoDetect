# RetroCheevoDetect

## Objetivo

Identificar momentos de exibição de notificações de retroachievements em vídeos de gameplay.

## Requisitos

- MacOS
- TensorFlow para MacOS e TensorFlow Metal

## Estrutura do projeto

RetroCheevoDetect/
├── data/
│   ├── train/
│   └── validation/
├── models/
├── scripts/
│   ├── train_model.py
│   └── analyze_video.py
└── README.md

## Configuração do Ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow-macos tensorflow-metal
```
## Instalação das dependências
```bash
pip install -r requirements.txt
```


## Treinamento do Modelo

Execute o script train_model.py para treinar o modelo.

```bash
python scripts/train_model.py
```

## Análise de Vídeo

Execute o script analyze_video.py para analisar um vídeo.

```bash
python scripts/analyze_video.py <video_path>
```