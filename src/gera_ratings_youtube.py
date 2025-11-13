# -*- coding: utf-8 -*-

import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


DATA_DIR = os.path.join("data", "youtube")
OUTPUT_PATH = os.path.join(DATA_DIR, "ratings.csv")

NUM_USERS = 500
MIN_INTERACTIONS = 60
MAX_INTERACTIONS = 120
RANDOM_STATE = 42

def fixar_semente(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

def gerar_ratings():
    print(">> Lendo vídeos do YouTube...")
    videos = pd.read_csv(os.path.join(DATA_DIR, "USvideos.csv"), encoding="latin1")
    categorias = videos["category_id"].unique().tolist()
    print(f"✅ {len(videos)} vídeos carregados em {len(categorias)} categorias.")

    registros = []
    fixar_semente()

    for uid in tqdm(range(1, NUM_USERS + 1), desc="Gerando interações simuladas de usuários"):
       
        prefs = random.sample(categorias, k=random.randint(2, 3))
        base_pref = np.random.dirichlet(np.ones(len(prefs)))
        base_pref = dict(zip(prefs, base_pref))

        n_interacoes = random.randint(MIN_INTERACTIONS, MAX_INTERACTIONS)

        for _ in range(n_interacoes):
            # Seleciona uma categoria com peso preferencial
            cat = random.choices(prefs, weights=list(base_pref.values()), k=1)[0]
            subset = videos[videos["category_id"] == cat]

            if subset.empty:
                continue

            # Aqui adicionamos um ruído (usuário pode assistir vídeos fora das preferências)
            if random.random() < 0.15:
                subset = videos.sample(min(100, len(videos)))

            # Selecionamos um vídeo aleatório dentro do subconjunto
            vid = subset.sample(1).iloc[0]["video_id"]

            # Gera nota entre 1 e 5 (colocamos que os usuários preferem bem-avaliar seus temas)
            nota = np.clip(np.random.normal(4.0 if cat in prefs else 2.5, 0.8), 1, 5)
            registros.append([uid, vid, round(nota, 1)])

    df = pd.DataFrame(registros, columns=["userId", "video_id", "rating"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n✅ ratings.csv gerado com sucesso: {OUTPUT_PATH}")
    print(f"Total de avaliações: {len(df)} | Usuários únicos: {df['userId'].nunique()}")
    print(f"Categorias simuladas: {sorted(random.sample(categorias, k=min(10, len(categorias))))}")

if __name__ == "__main__":
    gerar_ratings()
