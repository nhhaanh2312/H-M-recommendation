from typing import Optional
import pandas as pd


def build_candidate_features(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    candidates: pd.DataFrame,
    val_start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Compute a set of features for (customer_id, article_id) candidate pairs.

    Returns a DataFrame with features ready for model input.
    This is a simplified, CPU-friendly reproduction of the feature steps.
    """
    if val_start is None:
        val_start = transactions['t_dat'].max()

    trans = transactions.copy()
    trans['t_dat'] = pd.to_datetime(trans['t_dat'])
    val_start = pd.to_datetime(val_start)
    # Normalize ID formats to strings
    if 'article_id' in trans.columns:
        trans['article_id'] = trans['article_id'].astype(str).str.zfill(10)
    if 'customer_id' in trans.columns:
        trans['customer_id'] = trans['customer_id'].astype(str)
    candidates = candidates.copy()
    if 'article_id' in candidates.columns:
        candidates['article_id'] = candidates['article_id'].astype(str).str.zfill(10)
    if 'customer_id' in candidates.columns:
        candidates['customer_id'] = candidates['customer_id'].astype(str)

    # User-level features
    user_price = trans.groupby('customer_id')['price'].agg(['mean', 'max', 'std']).reset_index()
    user_price.columns = ['customer_id', 'user_mean_price', 'user_max_price', 'user_std_price']

    user_last = trans.groupby('customer_id')['t_dat'].max().reset_index()
    user_last['user_recency'] = (val_start - user_last['t_dat']).dt.days.astype('int32')
    user_last = user_last[['customer_id', 'user_recency']]

    user_total = trans.groupby('customer_id').size().reset_index(name='user_total_purchases')
    user_unique = trans.groupby('customer_id')['article_id'].nunique().reset_index(name='user_unique_items')

    # Item-level features
    item_pop = trans.groupby('article_id').size().reset_index(name='item_total_sales')

    item_sales_7d = trans[trans['t_dat'] >= (val_start - pd.Timedelta(days=7))].groupby('article_id').size().reset_index(name='item_sales_7d')
    item_sales_14d = trans[trans['t_dat'] >= (val_start - pd.Timedelta(days=14))].groupby('article_id').size().reset_index(name='item_sales_14d')
    item_sales_28d = trans[trans['t_dat'] >= (val_start - pd.Timedelta(days=28))].groupby('article_id').size().reset_index(name='item_sales_28d')

    item_price = trans.groupby('article_id')['price'].mean().reset_index(name='item_current_price')
    item_first_seen = trans.groupby('article_id')['t_dat'].min().reset_index()
    item_first_seen['item_age_days'] = (val_start - item_first_seen['t_dat']).dt.days.astype('int32')
    item_first_seen = item_first_seen[['article_id', 'item_age_days']]

    # User-item features
    user_item_rep = trans.groupby(['customer_id', 'article_id']).size().reset_index(name='user_item_purchase_count')
    user_item_last = trans.groupby(['customer_id', 'article_id'])['t_dat'].max().reset_index()
    user_item_last['user_item_recency'] = (val_start - user_item_last['t_dat']).dt.days.astype('int32')
    user_item_last = user_item_last[['customer_id', 'article_id', 'user_item_recency']]

    # Temporal decay
    tmp = trans.copy()
    tmp['days_ago'] = (val_start - tmp['t_dat']).dt.days.astype('float32')
    tmp['weight'] = (1.0 / (tmp['days_ago'] + 1)).astype('float32')
    item_weighted_pop = tmp.groupby('article_id')['weight'].sum().reset_index(name='item_weighted_pop')
    user_weighted_act = tmp.groupby('customer_id')['weight'].sum().reset_index(name='user_weighted_activity')
    user_item_weighted = tmp.groupby(['customer_id', 'article_id'])['weight'].sum().reset_index(name='user_item_weighted_score')

    # Merge features onto candidates
    feats = candidates.copy()
    feats = feats.merge(user_price, on='customer_id', how='left')
    feats = feats.merge(user_last, on='customer_id', how='left')
    feats = feats.merge(user_total, on='customer_id', how='left')
    feats = feats.merge(user_unique, on='customer_id', how='left')

    feats = feats.merge(item_pop, on='article_id', how='left')
    feats = feats.merge(item_sales_7d, on='article_id', how='left')
    feats = feats.merge(item_sales_14d, on='article_id', how='left')
    feats = feats.merge(item_sales_28d, on='article_id', how='left')
    feats = feats.merge(item_price, on='article_id', how='left')
    feats = feats.merge(item_first_seen, on='article_id', how='left')

    feats = feats.merge(user_item_rep, on=['customer_id', 'article_id'], how='left')
    feats = feats.merge(user_item_last, on=['customer_id', 'article_id'], how='left')

    feats = feats.merge(item_weighted_pop, on='article_id', how='left')
    feats = feats.merge(user_weighted_act, on='customer_id', how='left')
    feats = feats.merge(user_item_weighted, on=['customer_id', 'article_id'], how='left')

    # Article metadata
    if 'article_id' in articles.columns:
        arts = articles.copy()
        if 'article_id' in arts.columns:
            arts['article_id'] = arts['article_id'].astype(str).str.zfill(10)
            feats = feats.merge(arts.add_prefix('art_'), left_on='article_id', right_on='art_article_id', how='left')

    # Fill na and create cross features
    feats = feats.fillna(0)
    feats['price_diff'] = feats['item_current_price'] - feats.get('user_mean_price', 0)
    feats['is_new_to_user'] = (feats.get('user_item_purchase_count', 0) == 0).astype('int8')
    feats['purchase_rate'] = feats.get('user_item_purchase_count', 0) / (feats.get('user_total_purchases', 0) + 1)

    return feats
