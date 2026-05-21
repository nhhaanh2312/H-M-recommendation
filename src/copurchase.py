import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch

class CopurchaseFeature:

    def __init__(self, df_items: pd.DataFrame, df_users: pd.DataFrame, transactions: pd.DataFrame, articles: pd.DataFrame):
        self.df_items = df_items.set_index('article_id') if 'article_id' in df_items.columns else df_items
        self.df_users = df_users.set_index('customer_id') if 'customer_id' in df_users.columns else df_users
        self.articles = articles.set_index('article_id') if 'article_id' in articles.columns else articles

        tr = transactions.copy()
        tr['t_dat'] = pd.to_datetime(tr['t_dat'])
        self.trans = tr
        self.max_date = tr['t_dat'].max()

        self._seq_prob = None
        self._group_seq_prob = None
        self._cluster_copurchase = None
        self._item_km_centers = None

    def precompute_sequence(self):
        tr_sorted = self.trans.sort_values(['customer_id', 't_dat']).copy()
        tr_sorted['next_article'] = tr_sorted.groupby('customer_id')['article_id'].shift(-1)
        tr_sorted = tr_sorted.dropna(subset=['next_article'])

        counts = tr_sorted.groupby(['article_id', 'next_article']).size().reset_index(name='cnt')
        totals = counts.groupby('article_id')['cnt'].sum().reset_index(name='total')
        counts = counts.merge(totals, on='article_id')
        counts['prob'] = counts['cnt'] / counts['total']
        self._seq_prob = counts.set_index(['article_id', 'next_article'])['prob'].to_dict()

        art_group = self.articles['product_group_name'].to_dict() if 'product_group_name' in self.articles.columns else {}
        tr_sorted['group'] = tr_sorted['article_id'].map(art_group)
        tr_sorted['next_group'] = tr_sorted['next_article'].map(art_group)
        tr_sorted = tr_sorted.dropna(subset=['group', 'next_group'])

        gcounts = tr_sorted.groupby(['group', 'next_group']).size().reset_index(name='cnt')
        gtotals = gcounts.groupby('group')['cnt'].sum().reset_index(name='total')
        gcounts = gcounts.merge(gtotals, on='group')
        gcounts['prob'] = gcounts['cnt'] / gcounts['total']
        self._group_seq_prob = gcounts.set_index(['group', 'next_group'])['prob'].to_dict()

    def precompute_cluster_copurchase(self):
        cust_cluster = self.df_users['customer_cluster_id'].to_dict() if 'customer_cluster_id' in self.df_users.columns else {}
        self.trans['user_cluster'] = self.trans['customer_id'].map(cust_cluster)
        tr = self.trans.dropna(subset=['user_cluster']).copy()
        tr['user_cluster'] = tr['user_cluster'].astype(int)

        art_ids = list(self.df_items.index)
        art_idx = {aid: i for i, aid in enumerate(art_ids)}
        item_vec_matrix = np.vstack(self.df_items['item_embed_vec'].values)

        tr['item_idx'] = tr['article_id'].map(art_idx)
        tr = tr.dropna(subset=['item_idx'])
        tr['item_idx'] = tr['item_idx'].astype(int)

        n_clusters = int(tr['user_cluster'].max()) + 1 if len(tr) > 0 else 0
        ones = np.ones(len(tr))
        W = csr_matrix((ones, (tr['user_cluster'].values, tr['item_idx'].values)), shape=(n_clusters, len(art_ids)))
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        W = W.multiply(1.0 / row_sums[:, None])
        self._cluster_copurchase = W.dot(item_vec_matrix).astype('float32')

    def _build_cluster_pop(self, days: int):
        cutoff = self.max_date - pd.Timedelta(days=days)
        tr_win = self.trans[self.trans['t_dat'] > cutoff].copy()
        cust_cluster = self.df_users['customer_cluster_id'].to_dict() if 'customer_cluster_id' in self.df_users.columns else {}
        tr_win['user_cluster'] = tr_win['customer_id'].map(cust_cluster)

        pop = (
            tr_win.dropna(subset=['user_cluster'])
            .groupby(['user_cluster', 'article_id'])
            .size()
            .reset_index(name='cnt')
        )
        totals = pop.groupby('user_cluster')['cnt'].sum().reset_index(name='total')
        pop = pop.merge(totals, on='user_cluster')
        pop['score'] = pop['cnt'] / pop['total']
        return pop.set_index(['user_cluster', 'article_id'])['score'].to_dict()

    def add_features(self, candidates: pd.DataFrame, batch_size: int = 50000) -> pd.DataFrame:
        if self._seq_prob is None:
            self.precompute_sequence()
        if self._cluster_copurchase is None:
            self.precompute_cluster_copurchase()

        cp7 = self._build_cluster_pop(7)
        cp14 = self._build_cluster_pop(14)
        cp30 = self._build_cluster_pop(30)

        last_bought = self.trans.sort_values('t_dat').groupby('customer_id')['article_id'].last().to_dict()
        art_group = self.articles['product_group_name'].to_dict() if 'product_group_name' in self.articles.columns else {}

        item_embed_np = np.vstack(self.df_items['item_embed_vec'].values)
        user_embed_np = np.vstack(self.df_users['user_embed_vec'].values)
        user_7d_np = np.vstack(self.df_users['user_vec_7d'].values)
        user_14d_np = np.vstack(self.df_users['user_vec_14d'].values)
        user_30d_np = np.vstack(self.df_users['user_vec_30d'].values)
        cluster_cop_np = self._cluster_copurchase
        item_km_np = self._item_km_centers.astype('float32') if self._item_km_centers is not None else None
        item_clus_arr = self.df_items['item_cluster_id'].values.astype('int32') if 'item_cluster_id' in self.df_items.columns else np.zeros(len(self.df_items), dtype='int32')
        user_clus_arr = self.df_users['customer_cluster_id'].values.astype('int32') if 'customer_cluster_id' in self.df_users.columns else np.zeros(len(self.df_users), dtype='int32')

        item_id_to_idx = {aid: i for i, aid in enumerate(self.df_items.index)}
        user_id_to_idx = {uid: i for i, uid in enumerate(self.df_users.index)}

        def batch_cosine_gpu(A_np, B_np, device=None):
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            A = torch.from_numpy(A_np).to(dev)
            B = torch.from_numpy(B_np).to(dev)
            res = ((A / A.norm(dim=1, keepdim=True).clamp(1e-9)) *
                   (B / B.norm(dim=1, keepdim=True).clamp(1e-9))).sum(dim=1)
            out = res.cpu().numpy().astype('float32')
            del A, B, res
            torch.cuda.empty_cache()
            return out

        results = []
        n = len(candidates)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cand = candidates.iloc[start:end].copy()
            cids = cand['customer_id'].values
            aids = cand['article_id'].values

            i_idx = np.array([item_id_to_idx.get(a, 0) for a in aids])
            u_idx = np.array([user_id_to_idx.get(c, 0) for c in cids])
            u_clus = user_clus_arr[u_idx].clip(0)

            I = item_embed_np[i_idx]
            U = user_embed_np[u_idx]
            U7 = user_7d_np[u_idx]
            U14 = user_14d_np[u_idx]
            U30 = user_30d_np[u_idx]

            cand['cosine_sim_to_history'] = batch_cosine_gpu(U, I)
            cand['cosine_sim_to_recent_7d'] = batch_cosine_gpu(U7, I)
            cand['cosine_sim_to_recent_14d'] = batch_cosine_gpu(U14, I)
            cand['cosine_sim_to_recent_30d'] = batch_cosine_gpu(U30, I)

            last_arts = [last_bought.get(c) for c in cids]
            la_idx = np.array([item_id_to_idx.get(a, 0) if a is not None else 0 for a in last_arts])
            L = item_embed_np[la_idx]
            L[np.array([a is None for a in last_arts])] = 0
            cand['nn_copurchase_score'] = batch_cosine_gpu(L, I)

            cand['is_next_in_sequence'] = np.array([
                self._seq_prob.get((la, a), 0.0) if la is not None else 0.0
                for la, a in zip(last_arts, aids)
            ], dtype='float32')

            last_groups = [art_group.get(la) for la in last_arts]
            item_groups = [art_group.get(a) for a in aids]
            cand['is_group_next_in_sequence'] = np.array([
                self._group_seq_prob.get((lg, ig), 0.0) if lg is not None else 0.0
                for lg, ig in zip(last_groups, item_groups)
            ], dtype='float32')

            if item_km_np is not None:
                ic = item_clus_arr[i_idx].clip(0)
                C = item_km_np[ic]
                cand['style_cluster_score'] = batch_cosine_gpu(U, C)
            else:
                cand['style_cluster_score'] = np.float32(0.0)

            CM = cluster_cop_np[u_clus]
            cand['cluster_copurchase_score'] = batch_cosine_gpu(CM, I)

            cand['cluster_pop_7d'] = np.array([cp7.get((c, a), 0.0) for c, a in zip(u_clus, aids)], dtype='float32')
            cand['cluster_pop_14d'] = np.array([cp14.get((c, a), 0.0) for c, a in zip(u_clus, aids)], dtype='float32')
            cand['cluster_pop_30d'] = np.array([cp30.get((c, a), 0.0) for c, a in zip(u_clus, aids)], dtype='float32')

            results.append(cand)

        out = pd.concat(results, ignore_index=True)
        return out
