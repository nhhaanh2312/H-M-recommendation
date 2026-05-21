import re
from typing import Optional
import pandas as pd

def clean_hm_articles(file_path: str, output_path: Optional[str] = 'articles_cleaned_full.csv') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if 'detail_desc' in df.columns:
        df['detail_desc'] = df['detail_desc'].fillna('no description')
    if 'article_id' in df.columns:
        df['article_id'] = df['article_id'].astype(str).str.zfill(10)

    paired_cols = [
        ('graphical_appearance_no', 'graphical_appearance_name'),
        ('colour_group_code', 'colour_group_name'),
        ('perceived_colour_value_id', 'perceived_colour_value_name'),
        ('perceived_colour_master_id', 'perceived_colour_master_name'),
    ]

    def impute_group(group, code_col, name_col):
        valid = group[group[code_col] != -1]
        if len(valid) == 0:
            return group
        mode_code = valid[code_col].mode().iloc[0]
        mode_name = valid[valid[code_col] == mode_code][name_col].iloc[0]
        group.loc[group[code_col] == -1, code_col] = mode_code
        group.loc[group[name_col] == 'Unknown', name_col] = mode_name
        return group

    for code_col, name_col in paired_cols:
        if code_col in df.columns and (df[code_col] == -1).sum() > 0:
            df = df.groupby('product_type_no', group_keys=False).apply(
                lambda g: impute_group(g, code_col, name_col)
            )

    if 'product_type_no' in df.columns and (df['product_type_no'] == -1).sum() > 0:
        if 'product_group_name' in df.columns:
            df = df.groupby('product_group_name', group_keys=False).apply(
                lambda g: impute_group(g, 'product_type_no', 'product_type_name')
            )

    def _clean_text(text):
        if not isinstance(text, str):
            return text
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    if 'prod_name' in df.columns:
        df['prod_name'] = df['prod_name'].apply(_clean_text)
    if 'detail_desc' in df.columns:
        df['detail_desc'] = df['detail_desc'].apply(_clean_text)

    if output_path:
        df.to_csv(output_path, index=False)

    return df
