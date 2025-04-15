import pandas as pd
import numpy as np
from graph_features import graph_features

def generate_features(train_addresses, test_addresses, transactions, transfers, swaps):
    all_addresses = list(set(train_addresses + test_addresses))

    def process_chain(df, from_col, value_col, chain_col):
        grouped = df.groupby([from_col, chain_col]).agg({
            value_col: 'sum'
        }).unstack(fill_value=0)
        grouped.columns = [f"{value_col.lower()}_{col[1].lower()}" for col in grouped.columns]
        return grouped

    tx_count = transactions.groupby(['FROM_ADDRESS', 'CHAIN'])['TX_HASH'].count().unstack(fill_value=0)
    tx_count.columns = [f"tx_count_{col.lower()}" for col in tx_count.columns]

    tx_value = process_chain(transactions, 'FROM_ADDRESS', 'VALUE', 'CHAIN')
    token_count = transfers.groupby(['From_Address', 'CHAIN'])['Contract_Address'].nunique().unstack(fill_value=0)
    token_count.columns = [f"tokens_unique_{col.lower()}" for col in token_count.columns]

    swap_count = swaps.groupby(['ORIGIN_FROM_ADDRESS', 'CHAIN'])['TX_HASH'].count().unstack(fill_value=0)
    swap_count.columns = [f"swap_count_{col.lower()}" for col in swap_count.columns]

    # Merge all
    df = pd.DataFrame(index=all_addresses)
    df = df.join(tx_count, how='left')
    df = df.join(tx_value, how='left')
    df = df.join(token_count, how='left')
    df = df.join(swap_count, how='left')
    df = df.fillna(0)

    df["is_multi_chain_user"] = ((df.filter(like="_ethereum") > 0).any(axis=1)) & ((df.filter(like="_base") > 0).any(axis=1))
    df["is_multi_chain_user"] = df["is_multi_chain_user"].astype(int)

    # Add graph features
    graph_df = graph_features(transactions, transfers)
    df = df.join(graph_df, how="left").fillna(0)

    return df
