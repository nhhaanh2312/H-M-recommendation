from typing import List
import pandas as pd
import numpy as np


def top_popular_articles(transactions: pd.DataFrame, top_k: int = 1000) -> List[str]:
    pop = (
        transactions['article_id']
        .value_counts()
        .head(top_k)
        .index
        .astype(str)
        .tolist()
    )
    return pop


def compute_micro_segment(customers: pd.DataFrame, top_postal_n: int = 500) -> pd.DataFrame:
    cust = customers.copy()
    cust['age'] = pd.to_numeric(cust.get('age'), errors='coerce')
    cust['age_bucket'] = pd.cut(cust['age'], bins=[0, 18, 25, 35, 45, 55, 65, 200], labels=False).fillna(-1).astype(int)
    top_postal = cust['postal_code'].value_counts().head(top_postal_n).index
    cust['postal_clean'] = cust['postal_code'].where(cust['postal_code'].isin(top_postal), 'OTHER')
    cust['micro_segment'] = cust['age_bucket'].astype(str) + '_' + cust['postal_clean'].astype(str)
    return cust


def generate_candidates_for_user(
    customer_id: str,
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    top_n: int = 1000,
) -> List[str]:
    tx = transactions.copy()
    tx['customer_id'] = tx['customer_id'].astype(str)
    tx['article_id'] = tx['article_id'].astype(str).str.zfill(10)

    user_trans = tx[tx['customer_id'] == str(customer_id)]
    seen = []
    if len(user_trans) > 0:
        recent = (
            user_trans.sort_values('t_dat', ascending=False)['article_id']
            .astype(str)
            .tolist()
        )
        seen = list(dict.fromkeys(recent))
        
    popular = tx['article_id'].value_counts().index.astype(str).tolist()

    cust_row = customers.copy()
    cust_row['customer_id'] = cust_row['customer_id'].astype(str)
    cust_row = cust_row[cust_row['customer_id'] == str(customer_id)]
    micro_pop = []
    if len(cust_row) == 1 and 'micro_segment' in customers.columns:
        seg = cust_row.iloc[0]['micro_segment']
        merged = tx.merge(customers[['customer_id', 'micro_segment']].assign(customer_id=lambda d: d['customer_id'].astype(str)), on='customer_id', how='left')
        seg_pop = (
            merged[merged['micro_segment'] == seg]['article_id']
            .value_counts()
            .index.astype(str)
            .tolist()
        )
        micro_pop = seg_pop

    combined = []
    for src in (seen, micro_pop, popular):
        for a in src:
            if a not in combined:
                combined.append(str(a))
            if len(combined) >= top_n:
                break
        if len(combined) >= top_n:
            break

    return combined[:top_n]


def generate_candidates_batch(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    top_k: int = 1000,
) -> pd.DataFrame:

    customers = customers.copy()
    customers['customer_id'] = customers['customer_id'].astype(str)
    if 'micro_segment' not in customers.columns:
        customers = compute_micro_segment(customers)

    transactions = transactions.copy()
    transactions['customer_id'] = transactions['customer_id'].astype(str)
    transactions['article_id'] = transactions['article_id'].astype(str).str.zfill(10)
    articles = articles.copy()
    if 'article_id' in articles.columns:
        articles['article_id'] = articles['article_id'].astype(str).str.zfill(10)

    rows = []
    popular_list = top_popular_articles(transactions, top_k=top_k)

    for cust in customers['customer_id'].astype(str).tolist():
        cand = generate_candidates_for_user(cust, transactions, articles, customers, top_n=top_k)
        if len(cand) < top_k:
            for a in popular_list:
                if a not in cand:
                    cand.append(a)
                if len(cand) >= top_k:
                    break

        rows.extend([(cust, a) for a in cand[:top_k]])

    return pd.DataFrame(rows, columns=['customer_id', 'article_id'])
