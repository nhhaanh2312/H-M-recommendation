from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd


TOPK_TOTAL = 1000
TOPK_HEURISTIC = 400
TOPK_ATTR = 150
TOPK_SEGMENT = 100
RECENT_DAYS = [3, 7, 14]
RECENT_WEIGHTS = [6.0, 5.0, 4.0]
COVIS_WEIGHT = 12.0
REPEAT_WEIGHT = 10.0
RECENT_HIST_WEIGHT = 8.0


def top_popular_articles(transactions: pd.DataFrame, top_k: int = 1000) -> List[str]:
    return (
        transactions['article_id']
        .value_counts()
        .head(top_k)
        .index
        .astype(str)
        .tolist()
    )


def compute_micro_segment(customers: pd.DataFrame, top_postal_n: int = 500) -> pd.DataFrame:
    cust = customers.copy()
    cust['age'] = pd.to_numeric(cust.get('age'), errors='coerce')
    cust['age_bucket'] = pd.cut(
        cust['age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=False
    ).fillna(-1).astype(int)
    top_postal = cust['postal_code'].value_counts().head(top_postal_n).index
    cust['postal_clean'] = cust['postal_code'].where(cust['postal_code'].isin(top_postal), 'OTHER')
    cust['micro_segment'] = cust['age_bucket'].astype(str) + '_' + cust['postal_clean'].astype(str)
    return cust


def _normalize_ids(df: pd.DataFrame, article_col: str = 'article_id', customer_col: str = 'customer_id') -> pd.DataFrame:
    df = df.copy()
    if article_col in df.columns:
        df[article_col] = df[article_col].astype(str).str.zfill(10)
    if customer_col in df.columns:
        df[customer_col] = df[customer_col].astype(str)
    return df


def build_segment_top_items(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    days: int = 30,
    top_n: int = 200,
) -> Dict[str, List[str]]:
    max_date = transactions['t_dat'].max()
    recent = transactions[transactions['t_dat'] > max_date - pd.Timedelta(days=days)].copy()
    if 'micro_segment' not in customers.columns:
        customers = compute_micro_segment(customers)

    segment_map = customers.set_index('customer_id')['micro_segment'].to_dict()
    recent['micro_segment'] = recent['customer_id'].map(segment_map)

    seg_counts = (
        recent.dropna(subset=['micro_segment'])
        .groupby(['micro_segment', 'article_id'])
        .size()
        .reset_index(name='cnt')
        .sort_values(['micro_segment', 'cnt'], ascending=[True, False])
    )
    return {
        seg: sub['article_id'].head(top_n).tolist()
        for seg, sub in seg_counts.groupby('micro_segment')
    }


def build_attr_top_items(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    days: int = 30,
    top_n: int = 50,
) -> Dict[str, List[str]]:
    max_date = transactions['t_dat'].max()
    recent = transactions[transactions['t_dat'] > max_date - pd.Timedelta(days=days)].copy()
    art_df = articles[['article_id', 'product_group_name', 'colour_group_code']].copy()
    recent = recent.merge(art_df, on='article_id', how='left')
    recent['attr_key'] = (
        recent['product_group_name'].astype(str) + '_' + recent['colour_group_code'].astype(str)
    )

    attr_counts = (
        recent.dropna(subset=['attr_key'])
        .groupby(['attr_key', 'article_id'])
        .size()
        .reset_index(name='cnt')
        .sort_values(['attr_key', 'cnt'], ascending=[True, False])
    )
    return {
        key: sub['article_id'].head(top_n).tolist()
        for key, sub in attr_counts.groupby('attr_key')
    }


def build_covisitation_top_items(
    transactions: pd.DataFrame,
    days: int = 14,
    top_n: int = 20,
) -> Dict[str, List[str]]:
    recent = transactions[transactions['t_dat'] > transactions['t_dat'].max() - pd.Timedelta(days=days)]
    coocc = defaultdict(Counter)
    for _, group in recent.groupby(['customer_id', 't_dat']):
        items = list(dict.fromkeys(group['article_id'].tolist()))
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                coocc[items[i]][items[j]] += 1
                coocc[items[j]][items[i]] += 1
    return {k: [item for item, _ in v.most_common(top_n)] for k, v in coocc.items()}


def generate_candidates_for_user(
    customer_id: str,
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    segment_top_items: Dict[str, List[str]],
    attr_top_items: Dict[str, List[str]],
    article_attr_key: Dict[str, str],
    coocc_top: Dict[str, List[str]],
    popular_all: List[str],
    pop_rank: Dict[str, int],
    top_k: int = TOPK_TOTAL,
) -> List[Dict[str, object]]:
    user_trans = transactions[transactions['customer_id'] == str(customer_id)].sort_values('t_dat')
    user_hist = user_trans['article_id'].tolist()
    user_counter = Counter(user_hist)
    history = list(dict.fromkeys(user_hist))

    scores = defaultdict(float)
    covis_set = set()

    for item_id, cnt in user_counter.items():
        if cnt > 1:
            scores[item_id] += REPEAT_WEIGHT

    for pos, item_id in enumerate(history[-30:][::-1]):
        scores[item_id] += RECENT_HIST_WEIGHT / (pos + 1)

    for item_id in history[-3:]:
        covis_items = coocc_top.get(item_id, [])
        for pos, other in enumerate(covis_items[:60]):
            if other not in covis_set:
                scores[other] += COVIS_WEIGHT / (pos + 1)
                covis_set.add(other)

    recent_pop = {
        d: transactions[transactions['t_dat'] > transactions['t_dat'].max() - pd.Timedelta(days=d)]['article_id'].value_counts().index.tolist()
        for d in RECENT_DAYS
    }
    for d, weight in zip(RECENT_DAYS, RECENT_WEIGHTS):
        for pos, item_id in enumerate(recent_pop[d][:200]):
            scores[item_id] += weight / (pos + 1)

    heur_ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], pop_rank.get(kv[0], 10**12)),
    )

    last_item = history[-1] if history else None
    attr_cands = []
    if last_item is not None:
        attr_key = article_attr_key.get(last_item)
        if attr_key is not None:
            attr_cands = attr_top_items.get(attr_key, [])

    seg_name = None
    if 'micro_segment' in customers.columns:
        seg = customers.loc[customers['customer_id'] == str(customer_id), 'micro_segment']
        if len(seg) == 1:
            seg_name = seg.iloc[0]
    seg_cands = segment_top_items.get(seg_name, []) if seg_name is not None else []

    out_features = []
    seen = set()

    for item_id, score in heur_ranked:
        if item_id not in seen:
            out_features.append(
                {
                    'customer_id': customer_id,
                    'article_id': item_id,
                    'candidate_score': float(score),
                    'is_covisitation': int(item_id in covis_set),
                    'is_attr': 0,
                    'is_segment': 0,
                }
            )
            seen.add(item_id)
            if len(out_features) == TOPK_HEURISTIC:
                break

    for item_id in attr_cands:
        if item_id not in seen:
            out_features.append(
                {
                    'customer_id': customer_id,
                    'article_id': item_id,
                    'candidate_score': 0.0,
                    'is_covisitation': 0,
                    'is_attr': 1,
                    'is_segment': 0,
                }
            )
            seen.add(item_id)
            if len(out_features) == TOPK_HEURISTIC + TOPK_ATTR:
                break

    for item_id in seg_cands:
        if item_id not in seen:
            out_features.append(
                {
                    'customer_id': customer_id,
                    'article_id': item_id,
                    'candidate_score': 0.0,
                    'is_covisitation': 0,
                    'is_attr': 0,
                    'is_segment': 1,
                }
            )
            seen.add(item_id)
            if len(out_features) == TOPK_HEURISTIC + TOPK_ATTR + TOPK_SEGMENT:
                break

    for item_id in popular_all:
        if item_id not in seen:
            out_features.append(
                {
                    'customer_id': customer_id,
                    'article_id': item_id,
                    'candidate_score': 0.0,
                    'is_covisitation': 0,
                    'is_attr': 0,
                    'is_segment': 0,
                }
            )
            seen.add(item_id)
            if len(out_features) == top_k:
                break

    for rank, row in enumerate(out_features, start=1):
        row['candidate_rank'] = rank

    return out_features[:top_k]


def generate_candidates_batch(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    top_k: int = TOPK_TOTAL,
) -> pd.DataFrame:
    customers = _normalize_ids(customers)
    transactions = _normalize_ids(transactions)
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    articles = _normalize_ids(articles)

    if 'micro_segment' not in customers.columns:
        customers = compute_micro_segment(customers)

    popular_all = top_popular_articles(transactions, top_k=top_k)
    pop_rank = {art: i for i, art in enumerate(popular_all)}
    segment_top_items = build_segment_top_items(transactions, customers)
    attr_top_items = build_attr_top_items(transactions, articles)
    article_attr_key = {
        row['article_id']: f"{row.get('product_group_name', '')}_{row.get('colour_group_code', '')}"
        for _, row in articles[['article_id', 'product_group_name', 'colour_group_code']].fillna('').iterrows()
    }
    coocc_top = build_covisitation_top_items(transactions)

    rows = []
    for customer_id in customers['customer_id'].astype(str).tolist():
        rows.extend(
            generate_candidates_for_user(
                customer_id=customer_id,
                transactions=transactions,
                customers=customers,
                segment_top_items=segment_top_items,
                attr_top_items=attr_top_items,
                article_attr_key=article_attr_key,
                coocc_top=coocc_top,
                popular_all=popular_all,
                pop_rank=pop_rank,
                top_k=top_k,
            )
        )

    return pd.DataFrame(rows, columns=[
        'customer_id',
        'article_id',
        'candidate_rank',
        'candidate_score',
        'is_covisitation',
        'is_attr',
        'is_segment',
    ])
