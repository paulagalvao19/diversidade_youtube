#Efeitos da Personalização Algorítmica na Diversidade de Exposição no YouTube

Este repositório reúne os códigos, dados e resultados do experimento desenvolvido no Trabalho de Conclusão de Curso

##  Objetivo
Investigar como diferentes estratégias de recomendação **Popularidade**, **Similaridade de Conteúdo** e **Diversidade (MMR)** afetam a diversidade e a cobertura de vídeos recomendados no YouTube.

##  Estrutura do projeto
- `src/`: scripts principais em Python (`gera_ratings_youtube.py` e `main_youtube.py`)
- `data/`: dataset base (`USvideos.csv`) e resultados gerados (`metrics.csv`)
- `results/`: gráficos e visualizações
- `docs/`: artigo em LaTeX e referências bibliográficas
- `requirements.txt`: dependências para reprodução do experimento

## Metodologia
O experimento foi implementado em Python 3.12 com as bibliotecas:
`pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib` e `tqdm`.

O script `main_youtube.py` executa três modelos:
1. **Popularidade** — baseia-se em engajamento global.
2. **Similaridade** — recomenda vídeos com base em perfis de conteúdo.
3. **Diversidade (MMR)** — reordena as recomendações para maximizar a variedade interna.

## Resultados principais
| Modelo | ILD | Cobertura |
|---------|-----|------------|
| Diversidade MMR | 0.0749 | 0.0455 |
| Popularidade | 0.0565 | 0.0007 |
| Similaridade | 0.0051 | 0.0119 |

O modelo **MMR** apresentou o melhor equilíbrio entre relevância e diversidade, aumentando substancialmente a pluralidade das recomendações.

## Referência
Base de dados pública: [USvideos.csv - Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new)

## Licença
Distribuído sob a licença MIT — sinta-se à vontade para usar e adaptar.
