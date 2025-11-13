
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = os.path.join("data", "youtube")
OUTPUT_DIR = "outputs"

TAMANHO_LISTA = 10
MMR_LAMBDA = 0.5
RANDOM_STATE = 42
MAX_USERS = 300  # limitar usuários para melhorar um pouco o desempenho. Se quiser mais rápido basta diminuir


def fixar_semente(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

def garantir_pastas():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def carregar_dados():
    print(">> Lendo dados do YouTube...")
    videos = pd.read_csv(os.path.join(DATA_DIR, "USvideos.csv"), encoding="latin1")
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"), encoding="utf-8")

    if "videoId" in ratings.columns:
        ratings.rename(columns={"videoId": "video_id"}, inplace=True)

    videos["item_idx"] = videos.index
    ratings = ratings[ratings["video_id"].isin(videos["video_id"])]

    print(f"✅ {len(videos)} vídeos | {len(ratings)} avaliações válidas | {ratings['userId'].nunique()} usuários únicos")
    return videos, ratings


def criar_matriz_features(videos):
    print(">> Criando matriz de características combinadas (categoria + canal + views + engajamento)...")

    #  categorias e canais como vetores binários
    le_cat = LabelEncoder()
    le_channel = LabelEncoder()
    cat_enc = le_cat.fit_transform(videos["category_id"].astype(str))
    ch_enc = le_channel.fit_transform(videos["channel_title"].astype(str))

    matriz_cat = np.zeros((len(videos), len(np.unique(cat_enc))))
    matriz_channel = np.zeros((len(videos), len(np.unique(ch_enc))))

    for i, (c, ch) in enumerate(zip(cat_enc, ch_enc)):
        matriz_cat[i, c] = 1.0
        matriz_channel[i, ch] = 1.0


    videos["views_log"] = np.log1p(videos["views"])
    videos["like_ratio"] = videos["likes"] / (videos["dislikes"] + 1)
    videos["like_ratio"] = np.log1p(videos["like_ratio"])

    matriz_views = videos["views_log"].to_numpy().reshape(-1, 1)
    matriz_like = videos["like_ratio"].to_numpy().reshape(-1, 1)

    # Combina todas com pesos equilibrados
    matriz_final = np.hstack([
        matriz_cat * 0.35,
        matriz_channel * 0.35,
        matriz_views * 0.15,
        matriz_like * 0.15
    ])

    normalize(matriz_final, norm="l2", axis=1, copy=False)
    print("✅ Matriz combinada criada com sucesso!")
    return matriz_final


# CÁLCULO ILD
def ild_para_lista(indices, matriz):
    if len(indices) <= 1:
        return 0.0
    feats = matriz[indices]
    sim = cosine_similarity(feats)
    dist = 1 - sim[np.triu_indices(len(sim), k=1)]
    return np.mean(dist) if len(dist) > 0 else 0.0


# MODELOS

def recomendar_popularidade(user_items, ranking_global, topn=TAMANHO_LISTA):
    return [i for i in ranking_global if i not in user_items][:topn]

def recomendar_similaridade(user_items, matriz, topn=TAMANHO_LISTA):
    if not user_items:
        return []
    perfil = np.mean(matriz[list(user_items)], axis=0)
    perfil /= np.linalg.norm(perfil) + 1e-9
    sims = matriz.dot(perfil)
    ordem = np.argsort(-sims)
    return [i for i in ordem if i not in user_items][:topn]

def recomendar_diversidade(user_items, matriz, topn=TAMANHO_LISTA, mmr_lambda=MMR_LAMBDA):
    if not user_items:
        return []
    perfil = np.mean(matriz[list(user_items)], axis=0)
    perfil /= np.linalg.norm(perfil) + 1e-9
    base_scores = matriz.dot(perfil)
    candidatos = np.argsort(-base_scores)
    selecionados = []
    while len(selecionados) < topn and len(candidatos) > 0:
        melhor_item, melhor_score = None, -1e9
        for i in candidatos:
            if i in user_items:
                continue
            if not selecionados:
                mmr_score = base_scores[i]
            else:
                max_sim = max(cosine_similarity([matriz[i]], matriz[selecionados])[0])
                mmr_score = mmr_lambda * base_scores[i] - (1 - mmr_lambda) * max_sim
            if mmr_score > melhor_score:
                melhor_score, melhor_item = mmr_score, i
        selecionados.append(melhor_item)
        candidatos = [c for c in candidatos if c != melhor_item]
    return selecionados


# EXPERIMENTO

def avaliar_modelos(videos, ratings, matriz):
    id2idx = dict(zip(videos["video_id"], videos["item_idx"]))
    usuarios = ratings["userId"].unique()[:MAX_USERS]

    pop_counts = ratings.groupby("video_id")["userId"].count().sort_values(ascending=False)
    popularidade_ordem = [id2idx[vid] for vid in pop_counts.index if vid in id2idx]

    resultados, cobertura = [], {"Popularidade": set(), "Similaridade": set(), "DiversidadeMMR": set()}

    for uid in tqdm(usuarios, desc="Gerando recomendações"):
        vistos_idx = [id2idx[v] for v in ratings.loc[ratings["userId"] == uid, "video_id"].values if v in id2idx]
        modelos = {
            "Popularidade": recomendar_popularidade(vistos_idx, popularidade_ordem),
            "Similaridade": recomendar_similaridade(vistos_idx, matriz),
            "DiversidadeMMR": recomendar_diversidade(vistos_idx, matriz)
        }
        for nome, recs in modelos.items():
            ild = ild_para_lista(recs, matriz)
            cobertura[nome].update(recs)
            resultados.append({"usuario": uid, "modelo": nome, "ILD": ild})

    df = pd.DataFrame(resultados)
    metricas = df.groupby("modelo")[["ILD"]].mean().reset_index()
    metricas["Cobertura"] = metricas["modelo"].apply(lambda m: len(cobertura[m]) / len(videos))
    return metricas


# GRÁFICO

def plotar_metricas(df, caminho):
    sns.set_theme(style="whitegrid")
    cores = ["#A3C4F3", "#F9D8A9"]
    plt.figure(figsize=(8, 5))
    df_melt = df.melt(id_vars="modelo", value_vars=["ILD", "Cobertura"], var_name="Métrica", value_name="Valor")
    sns.barplot(x="modelo", y="Valor", hue="Métrica", data=df_melt, palette=cores, edgecolor="black")
    plt.title("Comparação entre modelos de recomendação (YouTube)", fontsize=13, weight="bold")
    plt.ylabel("Valor médio"); plt.xlabel("")
    plt.legend(title="Métrica", loc="upper right", fontsize=9)
    plt.tight_layout(); plt.savefig(caminho, dpi=250, bbox_inches="tight"); plt.close()


# MAIN

def main():
    fixar_semente(); garantir_pastas()
    videos, ratings = carregar_dados()
    matriz = criar_matriz_features(videos)
    print(">> Rodando experimento...")
    df_metricas = avaliar_modelos(videos, ratings, matriz)
    df_metricas.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    plotar_metricas(df_metricas, os.path.join(OUTPUT_DIR, "metrics.png"))
    print("\n✅ Experimento concluído!\n", df_metricas)
    print(f"Resultados salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
