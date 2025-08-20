import pandas as pd
import dask.dataframe as dd
import numpy as np

# 🧱 Дерево категорий
df_cat = pd.read_parquet(r"E:\PythonProjects\OZON_BCSS\ml_ozon_recsys_train_final_categories_tree\part-00000-163f9108-23d6-4fd6-98c4-7bcb73ee27e2-c000.snappy.parquet")
df_cat["depth"] = df_cat["ids"].apply(lambda x: len(x) - 1)
df_cat["is_leaf"] = df_cat["ids"].apply(lambda x: x[1] == -1 if len(x) > 1 else True)
print(df_cat[["catalogid", "depth", "is_leaf"]].head())

# 📦 История заказов
df_orders = dd.read_parquet(r"E:\PythonProjects\OZON_BCSS\ml_ozon_recsys_train_final_apparel_orders_data")
df_orders["timestamp"] = dd.to_datetime(df_orders["created_timestamp"])
df_orders["dayofweek"] = df_orders["timestamp"].dt.dayofweek
df_orders["hour"] = df_orders["timestamp"].dt.hour
print(df_orders[["item_id", "user_id", "dayofweek", "hour"]].head())

# 🧵 Каталог товаров
df_items = dd.read_parquet(r"E:\PythonProjects\OZON_BCSS\ml_ozon_recsys_train_final_apparel_items_data")
df_items["clip_vector"] = df_items["fclip_embed"].apply(lambda x: np.array(x), meta=('fclip_embed', 'object'))
print(df_items[["item_id", "clip_vector"]].head())

# 👥 Взаимодействие пользователей
df_interactions = dd.read_parquet(r"E:\PythonProjects\OZON_BCSS\ml_ozon_recsys_train_final_apparel_tracker_data")
df_interactions = df_interactions.rename(columns={"action_type": "event_type"})

# 🔍 Посмотрим реальные типы событий
print("Типы событий:")
print(df_interactions["event_type"].value_counts().compute())

# 🔄 Маппинг событий
event_mapping = {
    "pdp.characteristics": "view",
    "cart.cartSplit": "add_to_cart",
    "pdp.richContent": "view"
}
df_interactions["event_type"] = df_interactions["event_type"].map(event_mapping, meta=('event_type', 'object'))

# 🧼 Очистка
df_interactions = df_interactions.dropna(subset=["user_id", "item_id", "event_type", "timestamp"])
valid_events = ["view", "add_to_cart"]
df_interactions = df_interactions[df_interactions["event_type"].isin(valid_events)]

# 🕒 Временные фичи
df_interactions["timestamp"] = dd.to_datetime(df_interactions["timestamp"])
df_interactions["dayofweek"] = df_interactions["timestamp"].dt.dayofweek
df_interactions["hour"] = df_interactions["timestamp"].dt.hour

# ⚖️ Веса событий
event_weights = {
    "view": 1.0,
    "add_to_cart": 3.0
}
df_interactions["weight"] = df_interactions["event_type"].map(event_weights, meta=('weight', 'float64'))


print("Пример взаимодействий:")
print(df_interactions[["user_id", "item_id", "event_type", "weight"]].head())

df_matrix = df_interactions.groupby(["user_id", "item_id"])["weight"].sum().reset_index()
df_matrix_pd = df_matrix.compute()
print("User-item матрица:")
print(df_matrix_pd.head())
