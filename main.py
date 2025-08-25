import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor

# 1) Подготовка данных (как у тебя)
import dask.dataframe as dd

# Дерево категорий
df_cat = pd.read_parquet(
    r"E:\PythonProjects\OZON_CLEAN\ml_ozon_recsys_train_final_categories_tree"
    r"\part-00000-163f9108-23d6-4fd6-98c4-7bcb73ee27e2-c000.snappy.parquet"
)
df_cat["depth"] = df_cat["ids"].apply(lambda x: len(x) - 1)
df_cat["is_leaf"] = df_cat["ids"].apply(lambda x: x[1] == -1 if len(x) > 1 else True)

# История заказов
df_orders = dd.read_parquet(
    r"E:\PythonProjects\OZON_CLEAN\ml_ozon_recsys_train_final_apparel_orders_data"
)
df_orders["timestamp"] = dd.to_datetime(df_orders["created_timestamp"])
df_orders = df_orders.sort_values(["user_id", "timestamp"])
df_orders_pd = df_orders[["user_id", "item_id", "timestamp"]].compute()

# Каталог товаров
df_items = dd.read_parquet(
    r"E:\PythonProjects\OZON_CLEAN\ml_ozon_recsys_train_final_apparel_items_data"
)
df_items["clip_vector"] = df_items["fclip_embed"].apply(
    lambda x: np.array(x), meta=("fclip_embed", "object")
)
df_items_pd = df_items[["item_id", "clip_vector"]].compute()

# Взаимодействия пользователей
df_inter = dd.read_parquet(
    r"E:\PythonProjects\OZON_CLEAN\ml_ozon_recsys_train_final_apparel_tracker_data"
).rename(columns={"action_type": "event_type"})
# Маппинг и фильтрация
event_mapping = {
    "pdp.characteristics": "view",
    "cart.cartSplit": "add_to_cart",
    "pdp.richContent": "view",
}
df_inter["event_type"] = df_inter["event_type"].map(
    event_mapping, meta=("event_type", "object")
)
df_inter = df_inter.dropna(
    subset=["user_id", "item_id", "event_type", "timestamp"]
)
valid_events = ["view", "add_to_cart"]
df_inter = df_inter[df_inter["event_type"].isin(valid_events)]
df_inter["timestamp"] = dd.to_datetime(df_inter["timestamp"])
df_inter = df_inter.compute()
df_inter["weight"] = df_inter["event_type"].map({"view": 1.0, "add_to_cart": 3.0})
df_matrix_pd = (
    df_inter.groupby(["user_id", "item_id"])["weight"].sum().reset_index()
)

# ==== 2) Функции обучения и сохранения моделей ====

# 2A) LightGBM для классификации is_leaf по depth
def train_lightgbm():
    X = df_cat[["depth"]].values
    y = df_cat["is_leaf"].astype(int).values
    dtrain = lgb.Dataset(X, y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=100)
    preds = (model.predict(X) > 0.5).astype(int)
    acc = accuracy_score(y, preds)
    # Сохраняем
    joblib.dump(model, "lightgbm_model.pkl")
    with open("lightgbm_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
    print("LightGBM done, acc =", acc)
    return acc

# 2B) GRU для последовательности заказов: предсказываем следующий item_id
class OrdersDataset(Dataset):
    def __init__(self, df, max_len=10):
        # создаём словарь item->idx
        items = df["item_id"].unique()
        self.item2idx = {it: i + 1 for i, it in enumerate(items)}
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self.max_len = max_len

        # формируем последовательности
        self.sequences = []
        for uid, g in df.groupby("user_id"):
            seq = [self.item2idx[it] for it in g["item_id"].tolist()]
            if len(seq) < 2:
                continue
            # разбиваем на пары (inp, target)
            for i in range(1, len(seq)):
                inp = seq[max(0, i - max_len) : i]
                pad = [0] * (max_len - len(inp))
                self.sequences.append((pad + inp, seq[i]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class GRUNet(nn.Module):
    def __init__(self, n_items, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_items + 1)

    def forward(self, x):
        x = self.emb(x)
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out

def train_gru():
    ds = OrdersDataset(df_orders_pd)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUNet(n_items=len(ds.item2idx)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Несколько эпох
    for epoch in range(3):
        total_loss = 0
        for seqs, targs in loader:
            seqs, targs = seqs.to(device), targs.to(device)
            opt.zero_grad()
            logits = model(seqs)
            loss = loss_fn(logits, targs)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"GRU epoch {epoch} loss {total_loss/len(loader):.4f}")

    # Сохраняем результаты
    torch.save(model.state_dict(), "gru_model.pt")
    with open("gru_results.txt", "w") as f:
        f.write(f"Final loss: {total_loss/len(loader):.4f}\n")
    print("GRU done")
    return total_loss / len(loader)

# 2C) KMeans по CLIP-эмбеддингам
def train_kmeans():
    X = np.vstack(df_items_pd["clip_vector"].values)
    kmeans = KMeans(n_clusters=10, random_state=42).fit(X)
    # Сохраняем метки
    df_items_pd["cluster"] = kmeans.labels_
    df_items_pd[["item_id", "cluster"]].to_csv("items_clusters.csv", index=False)
    joblib.dump(kmeans, "kmeans_model.pkl")
    print("KMeans done")
    return None

# 2D) Neural Collaborative Filtering (NCF) для user-item матрицы
class NCFDataset(Dataset):
    def __init__(self, df, num_neg=1):
        self.users = df["user_id"].unique()
        self.items = df["item_id"].unique()
        self.user2idx = {u: i for i, u in enumerate(self.users)}
        self.item2idx = {i: j for j, i in enumerate(self.items)}

        # positive pairs
        pos = [(self.user2idx[u], self.item2idx[i]) for u, i in zip(df["user_id"], df["item_id"])]
        neg = []
        np.random.seed(42)
        for (u, _) in pos:
            for _ in range(num_neg):
                neg_i = np.random.randint(len(self.items))
                neg.append((u, neg_i))
        self.pairs = pos + neg
        self.labels = [1] * len(pos) + [0] * len(neg)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(self.labels[idx], dtype=torch.float32)

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.u_emb = nn.Embedding(n_users, emb_dim)
        self.i_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, u, i):
        xu = self.u_emb(u)
        xi = self.i_emb(i)
        x = torch.cat([xu, xi], dim=-1)
        return self.mlp(x).squeeze()

def train_ncf():
    ds = NCFDataset(df_matrix_pd, num_neg=1)
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(len(ds.users), len(ds.items)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(3):
        total_loss = 0
        for u, i, y in loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            opt.zero_grad()
            logits = model(u, i)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"NCF epoch {epoch} loss {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "ncf_model.pt")
    with open("ncf_results.txt", "w") as f:
        f.write(f"Final loss: {total_loss/len(loader):.4f}\n")
    print("NCF done")
    return total_loss / len(loader)

# ==== 3) Параллельный запуск ====
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as exe:
        exe.submit(train_lightgbm)
        exe.submit(train_gru)
        exe.submit(train_kmeans)
        exe.submit(train_ncf)

    print("All models trained and saved.")
