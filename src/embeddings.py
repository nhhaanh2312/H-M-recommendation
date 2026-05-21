import os
import numpy as np
import pandas as pd
import torch
import clip
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from typing import Optional, List, Tuple

class Embedding:
    def __init__(
        self,
        articles: pd.DataFrame,
        transactions: pd.DataFrame,
        customers: pd.DataFrame,
        image_dir: Optional[str] = None,
        n_item_clusters: int = 50,
        n_user_clusters: int = 30,
        device: str = "cuda",
        out_dir: str = ".",
    ):
        self.articles = articles.copy()
        self.trans = transactions.copy()
        self.customers = customers.copy()
        self.image_dir = image_dir
        self.n_item_clus = n_item_clusters
        self.n_user_clus = n_user_clusters
        self.device = device
        self.out_dir = out_dir
        
        if 'article_id' in self.articles.columns:
            self.articles['article_id'] = self.articles['article_id'].astype(str).str.zfill(10)
        if 'article_id' in self.trans.columns:
            self.trans['article_id'] = self.trans['article_id'].astype(str).str.zfill(10)

        self.item_embed: Optional[np.ndarray] = None
        self.art_id_list: List[str] = self.articles['article_id'].tolist()
        self.item_km_centers = None

    def _build_text_vec(self) -> np.ndarray:
        def build_text(row):
            parts = [
                f"Product: {row.get('prod_name','')}",
                f"Type: {row.get('product_type_name','')}",
                f"Group: {row.get('product_group_name','')}",
                f"Appearance: {row.get('graphical_appearance_name','')}",
                f"Colour: {row.get('perceived_colour_master_name','')} {row.get('colour_group_name','')}",
                f"Department: {row.get('department_name','')}",
                f"Section: {row.get('section_name','')}",
                f"Garment: {row.get('garment_group_name','')}",
                f"Index: {row.get('index_group_name','')} {row.get('index_name','')}",
                f"Description: {row.get('detail_desc','')}",
            ]
            return " | ".join(p for p in parts if p and p.split(": ", 1)[1].strip())

        texts = self.articles.apply(build_text, axis=1).tolist()
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        raw = model.encode(texts, batch_size=256, show_progress_bar=False, device=self.device)
        svd = TruncatedSVD(n_components=96, random_state=42)
        return svd.fit_transform(raw)

    def _build_image_vec(self) -> np.ndarray:
        if self.image_dir is None:
            return np.zeros((len(self.art_id_list), 64), dtype=np.float32)
        model_clip, preprocess = clip.load('ViT-B/32', device=self.device)

        class _DS(Dataset):
            def __init__(self, ids, img_dir, prep):
                self.ids = ids
                self.dir = img_dir
                self.prep = prep

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, idx):
                aid = str(self.ids[idx]).zfill(10)
                folder = aid[:3]
                path = os.path.join(self.dir, folder, f"{aid}.jpg")
                try:
                    img = preprocess(Image.open(path).convert('RGB'))
                except Exception:
                    img = torch.zeros(3, 224, 224)
                return img, idx

        ds = _DS(self.art_id_list, self.image_dir, preprocess)
        loader = DataLoader(ds, batch_size=256, num_workers=2, pin_memory=True)
        raw = np.zeros((len(self.art_id_list), 512), dtype=np.float32)

        model_clip.eval()
        with torch.no_grad():
            for imgs, idxs in loader:
                feats = model_clip.encode_image(imgs.to(self.device))
                raw[idxs.numpy()] = feats.cpu().float().numpy()

        svd = TruncatedSVD(n_components=64, random_state=42)
        return svd.fit_transform(raw)

    def build_item_embeddings(self) -> pd.DataFrame:
        text_vec = self._build_text_vec()
        image_vec = self._build_image_vec()

        combined = np.hstack([text_vec, image_vec])
        self.item_embed = normalize(combined)

        km = KMeans(n_clusters=self.n_item_clus, random_state=42, n_init=10)
        item_cluster = km.fit_predict(self.item_embed)
        self.item_km_centers = km.cluster_centers_

        df_items = self.articles[['article_id']].copy()
        df_items['item_embed_vec'] = list(self.item_embed)
        df_items['item_cluster_id'] = item_cluster.astype('int16')

        out = os.path.join(self.out_dir, 'df_items.parquet')
        df_items.to_parquet(out, index=False)
        return df_items

    def build_user_embeddings(self, df_items: pd.DataFrame) -> pd.DataFrame:
        tr = self.trans.copy()
        tr['t_dat'] = pd.to_datetime(tr['t_dat'])
        max_date = tr['t_dat'].max()
        tr['days_since'] = (max_date - tr['t_dat']).dt.days + 1

        dim = self.item_embed.shape[1]
        item_vec_matrix = np.vstack(df_items['item_embed_vec'].values)
        art_idx = {aid: i for i, aid in enumerate(df_items['article_id'])}

        tr['item_idx'] = tr['article_id'].map(art_idx)
        tr = tr.dropna(subset=['item_idx'])
        tr['item_idx'] = tr['item_idx'].astype(int)

        customers = tr['customer_id'].unique()
        cust_idx = {c: i for i, c in enumerate(customers)}
        tr['cust_idx'] = tr['customer_id'].map(cust_idx)
        n_users = len(customers)

        def _vectorized_weighted_mean(mask=None):
            sub = tr if mask is None else tr[mask]
            if len(sub) == 0:
                return np.zeros((n_users, dim), dtype=np.float32)

            weights = 1.0 / sub['days_since'].values
            W = csr_matrix((weights, (sub['cust_idx'].values, sub['item_idx'].values)), shape=(n_users, len(art_idx)))
            row_sums = np.asarray(W.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1
            W = W.multiply(1.0 / row_sums[:, None])
            return W.dot(item_vec_matrix)

        user_vecs = _vectorized_weighted_mean()
        user_vecs_7d = _vectorized_weighted_mean(tr['days_since'] <= 7)
        user_vecs_14d = _vectorized_weighted_mean(tr['days_since'] <= 14)
        user_vecs_30d = _vectorized_weighted_mean(tr['days_since'] <= 30)

        df_users = pd.DataFrame({
            'customer_id': customers,
            'user_embed_vec': list(user_vecs),
            'user_vec_7d': list(user_vecs_7d),
            'user_vec_14d': list(user_vecs_14d),
            'user_vec_30d': list(user_vecs_30d),
        })

        demo_cols = [c for c in ['customer_id', 'FN', 'Active', 'club_member_status', 'fashion_news_frequency', 'age'] if c in self.customers.columns]
        if len(demo_cols) > 0:
            demo = self.customers[demo_cols].copy()
            df_users = df_users.merge(demo, on='customer_id', how='left')

        vecs = np.vstack(df_users['user_embed_vec'].values)
        km = KMeans(n_clusters=self.n_user_clus, random_state=42, n_init=10)
        df_users['customer_cluster_id'] = km.fit_predict(vecs).astype('int16')
        self.user_km_centers = km.cluster_centers_

        out = os.path.join(self.out_dir, 'df_users.parquet')
        df_users.to_parquet(out, index=False)
        return df_users

    def set_item_km_centers(self, centers: np.ndarray):
        self.item_km_centers = centers

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_items = self.build_item_embeddings()
        df_users = self.build_user_embeddings(df_items)
        return df_items, df_users
