import argparse
import os
import pandas as pd
from data_preprocessing import clean_hm_articles
from retrieve import compute_micro_segment, generate_candidates_batch
from feature_engineering import build_candidate_features
from rerank import predict_ensemble, load_lgbm_models, load_cat_models


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out', type=str, default='output')

    parser.add_argument('--top_k', type=int, default=1000)
    parser.add_argument('--val_start', type=str, default=None)

    parser.add_argument('--lgbm', nargs='*', default=None)
    parser.add_argument('--cat', nargs='*', default=None)

    return parser.parse_args()


def run_pipeline(args):

    os.makedirs(args.out, exist_ok=True)

    articles_path = os.path.join(args.data_dir, 'data/articles.csv')
    transactions_path = os.path.join(args.data_dir, 'data/transactions_train.csv')
    customers_path = os.path.join(args.data_dir, 'data/customers.csv')

    articles = clean_hm_articles(articles_path, output_path=None)
    
    trans = pd.read_csv(
        transactions_path,
        dtype={'customer_id': str, 'article_id': str}
    )
    trans['t_dat'] = pd.to_datetime(trans['t_dat'])
    
    customers = pd.read_csv(
        customers_path,
        dtype={'customer_id': str}
    )

    customers = compute_micro_segment(customers)

    candidates = generate_candidates_batch(
        customers,
        trans,
        articles,
        top_k=args.top_k
    )

    candidates.to_csv(
        os.path.join(args.out, 'data/candidates.csv'),
        index=False
    )

    feats = build_candidate_features(
        trans,
        articles,
        customers,
        candidates,
        val_start=args.val_start
    )

    feats.to_parquet(
        os.path.join(args.out, 'data/features_candidates.parquet'),
        index=False
    )

    if args.lgbm or args.cat:

        lgbm_models = (
            load_lgbm_models(args.lgbm)
            if args.lgbm else None
        )

        cat_models = (
            load_cat_models(args.cat)
            if args.cat else None
        )

        ignore_cols = ['customer_id', 'article_id']
        X = feats.drop(
            columns=[c for c in ignore_cols if c in feats.columns]
        )

        feats['score'] = predict_ensemble(
            lgbm_models=lgbm_models,
            cat_models=cat_models,
            X=X
        )

        top12 = (
            feats
            .sort_values(
                ['customer_id', 'score'],
                ascending=[True, False]
            )
            .groupby('customer_id')
            .head(12)
            .groupby('customer_id')['article_id']
            .apply(lambda arr: ' '.join(map(str, arr)))
            .reset_index()
        )

        top12.to_csv(
            os.path.join(args.out, 'submission.csv'),
            index=False
        )



args = parse_args()
run_pipeline(args)