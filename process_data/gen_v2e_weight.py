import pandas as pd
import torch
from datetime import datetime, timedelta

def convert_e_list_to_v_list(e_list):
    max_vertex = max(v for e in e_list for v in e)
    v_list = [[] for _ in range(max_vertex + 1)]

    for e_idx, vertices in enumerate(e_list):
        for v in vertices:
            v_list[v].append(e_idx)

    return v_list


def generate_v2e_weight(
    vertex_info_path: str,
    hyperedge_info_path: str,
    order_info_path: str,
    dataset: dict, 
    start_date: datetime, 
    series_len: int,
    e_list: list, 
    
) -> dict:

    vertex_df = pd.read_csv(vertex_info_path)
    hyperedge_df = pd.read_csv(hyperedge_info_path)
    orders_df = pd.read_csv(order_info_path)

    vendor_to_index = dict(zip(vertex_df['vender_id'], vertex_df['vertex_index']))
    city_to_index = dict(zip(hyperedge_df['content'], hyperedge_df['index']))

    orders_df['send_tm'] = pd.to_datetime(orders_df['send_tm'])
    end_date = start_date + timedelta(series_len - 1)
    mask = (orders_df['send_tm'] >= start_date) & (orders_df['send_tm'] <= end_date)
    orders_df = orders_df[mask]

    order_volume_single_city = {}
    order_volume_total = {} 
    print('Start processing orders...')
    for _, row in orders_df.iterrows():
        vender_id = row['vender_id']
        city = row['city']
        order_num = float(row['order_num'])

        if vender_id not in vendor_to_index or city not in city_to_index:
            continue

        hyperedge_id = city_to_index[city]
        vertex_id = vendor_to_index[vender_id]

        order_volume_single_city[(hyperedge_id, vertex_id)] = order_volume_single_city.get((hyperedge_id, vertex_id), 0.0) + order_num
        order_volume_total[vertex_id] = order_volume_total.get(vertex_id, 0.0) + order_num

    # ---------- generate sparse matrix ----------
    m_list, n_list, val_list = [], [], []
    
    v_list = convert_e_list_to_v_list(e_list=e_list)

    for vertex_id, vertex in enumerate(v_list):
        for hyperedge_id in vertex:
            if (hyperedge_id, vertex_id) in order_volume_single_city:
                value = order_volume_single_city[(hyperedge_id, vertex_id)]
                total = order_volume_total.get(vertex_id, 0.0)
                m_list.append(hyperedge_id)
                n_list.append(vertex_id)
                val_list.append(value / total)
            else:
                m_list.append(hyperedge_id)
                n_list.append(vertex_id)
                val_list.append(0)


    indices = torch.tensor([m_list, n_list], dtype=torch.long)
    values = torch.tensor(val_list, dtype=torch.float32)
    M = hyperedge_df.shape[0]
    N = vertex_df.shape[0]
    P = torch.sparse_coo_tensor(indices, values, size=(M, N))
    dataset['v2e_weight'] = P
    return dataset