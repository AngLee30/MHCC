import os
import pickle
import torch
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

from gen_v2e_weight import generate_v2e_weight

def filter_venders_by_active_days(
    data_root_dir: str,
    output_file: str,
    start_date: datetime,
    series_len: int,
    non_zero_days: int,
    filter_file: str = None,
    bucket_num: int = 500
):
    '''
    Extract, from all buckets, all vendors that satisfy the following conditions:

    - Starting from `start_date`, within any consecutive `series_len` days, there are at least `non_zero_days` days on which the order volume is non-zero; and  
    - In the 7 days following this period, the total order volume is non-zero.

    If a `filter_file` (containing a `vender_id` column) is provided, the vendors listed in this file will be excluded.
    '''

    exclude_venders = set()
    if filter_file is not None:
        filter_df = pd.read_csv(filter_file)
        if 'vender_id' in filter_df.columns:
            exclude_venders = set(filter_df['vender_id'].unique())
        else:
            raise ValueError(f"{filter_file} must contain a 'vender_id' column.")

    check_end_date = start_date + timedelta(days=series_len-1)
    tail_start_date = start_date + timedelta(days=series_len)
    tail_end_date = start_date + timedelta(days=series_len+6)

    result = []
    print('Start processing buckets...')

    for i in range(bucket_num):
        file_path = os.path.join(data_root_dir, f"bucket_{i}.csv")
        df = pd.read_csv(file_path, parse_dates=["send_tm"])
        df = df[(df["send_tm"] >= start_date) & (df["send_tm"] <= tail_end_date)]

        if df.empty:
            continue

        for vender_id, group in df.groupby("vender_id"):
            if vender_id in exclude_venders:
                continue

            check_range = group[(group["send_tm"] >= start_date) & (group["send_tm"] <= check_end_date)]
            if check_range.empty:
                continue

            daily_sum = (
                check_range.groupby(check_range["send_tm"].dt.date)["order_num"].sum().reset_index(name="daily_order_sum")
            )
            active_days = (daily_sum["daily_order_sum"] > 0).sum()

            if active_days >= non_zero_days:
                tail_range = group[(group["send_tm"] >= tail_start_date) & (group["send_tm"] <= tail_end_date)]
                tail_order_sum = tail_range["order_num"].sum()

                if tail_order_sum > 0:
                    full_range = group[(group["send_tm"] >= start_date) & (group["send_tm"] <= tail_end_date)]
                    result.append(full_range)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} buckets.")

    pd.concat(result).to_csv(output_file, index=False)


def filter_venders_by_daily_avg(
        input_csv: str, 
        output_csv: str, 
        threshold: float,
        start_date: datetime,
        series_len: int
    ) -> pd.DataFrame:
    '''
    Read the input_csv and retain all rows for vendors whose 30-day average daily order volume is below `threshold`.
    '''
    df = pd.read_csv(input_csv, parse_dates=['send_tm']).sort_values(by=['vender_id', 'send_tm'])

    end_date = start_date + pd.Timedelta(days=series_len-1)
    df_series = df[(df['send_tm'] >= start_date) & (df['send_tm'] <= end_date)]

    daily_avg = df_series.groupby('vender_id')['order_num'].sum() / series_len

    valid_venders = daily_avg[daily_avg < threshold].index

    filtered_df = df[df['vender_id'].isin(valid_venders)].reset_index(drop=True)
    filtered_df[['vender_id', 'city', 'send_tm', 'order_num']].to_csv(output_csv, index=False)


def count_element(
        input_path: str,
        output_path: str,
        task: str,
    ):
    df = pd.read_csv(input_path, dtype=str)

    if task == "industry":
        df = pd.read_csv(input_path)
     
        df['industry_lv2_full'] = df['industry_lv1'] + '*' + df['industry_lv2']
        df['industry_lv3_full'] = df['industry_lv1'] + '*' + df['industry_lv2'] + '*' + df['industry_lv3']
        lv1_count = df['industry_lv1'].dropna().nunique() # number of level-1 industry
        lv2_count = df['industry_lv2_full'].dropna().nunique() # number of level-2 industry
        lv3_count = df['industry_lv3_full'].dropna().nunique() # number of level-3 industry

        print(f"number of level-1 industries: {lv1_count}")
        print(f"number of level-2 industries: {lv2_count}")
        print(f"number of level-3 industries: {lv3_count}")

        lv1_set = df['industry_lv1'].dropna().drop_duplicates()
        lv2_set = df['industry_lv2_full'].dropna().drop_duplicates()
        lv3_set = df['industry_lv3_full'].dropna().drop_duplicates()

        all_levels = pd.concat([lv1_set, lv2_set, lv3_set], ignore_index=True).drop_duplicates()

        result_df = pd.DataFrame({'content': all_levels})
        result_df['index'] = range(len(result_df))
        result_df.to_csv(output_path, index=False, encoding='utf-8')
    
    elif task == "city":
        unique_cities = df['city'].nunique()
        print(f"number of cities: {unique_cities}")

        df['content'] = df['city']
        city_df = df['content'].drop_duplicates().sort_values().reset_index(drop=True)
        city_df = pd.DataFrame({'content': city_df})
        city_df['index'] = city_df.index

        city_df.to_csv(output_path, index=False, encoding='utf-8-sig')


def extract_order_series(order_file: str, start_date: datetime, series_len: int) -> dict:

    df = pd.read_csv(order_file, parse_dates=['send_tm'])
    df = df[['vender_id', 'send_tm', 'order_num']].sort_values(by=['vender_id', 'send_tm'])

    end_date = start_date + timedelta(days = series_len + 6)

    result_dict = {}
    current_vender = None
    current_df = []

    for _, row in df.iterrows():
        vender_id = row['vender_id']
        send_tm = row['send_tm']
        order_num = row['order_num']

        if vender_id != current_vender:
            if current_vender is not None:
                result_dict[current_vender] = get_single_vendor_order_volume(current_df, start_date, series_len)
            current_vender = vender_id
            current_df = []
        
        if start_date <= send_tm <= end_date:
            current_df.append((send_tm, order_num))

    if current_vender is not None:
        result_dict[current_vender] = get_single_vendor_order_volume(current_df, start_date, series_len)

    return result_dict


def get_single_vendor_order_volume(data, start_date, series_len):
    day_order_sum = {start_date + timedelta(days=i): 0 for i in range(series_len + 7)}

    for date, order_num in data:
        if date in day_order_sum:
            day_order_sum[date] += order_num

    series_days = [day_order_sum[start_date + timedelta(days=i)] for i in range(series_len)]
    sum_7 = sum(day_order_sum[start_date + timedelta(days=i)] for i in range(series_len, series_len + 7))

    return series_days + [sum_7]


def random_true_false_pair(length: int, proportion = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    num_true = int(length * proportion)
    num_false = length - num_true

    values = torch.tensor([True] * num_true + [False] * num_false, dtype=torch.bool)
    permuted = values[torch.randperm(length)]
    complement = ~permuted
    return permuted, complement


def feature_to_tensor(
        vertex_path: str,
        order_volume: dict, 
        proportion: float = 0.8, # train–val split ratio
        val_file: str = None # specify some `vender_id` as the validation set
    ):
    df = pd.read_csv(vertex_path)['vender_id']

    Feature = []
    Ground_truth = []

    for row in df:
        vendor_order_volume = order_volume[row]
        feature = vendor_order_volume[:-1]
        ground_truth = vendor_order_volume[-1]
        Feature.append(feature)
        Ground_truth.append(ground_truth)

    X = torch.tensor(Feature, dtype=torch.float32)       # [N, 30]
    Y = torch.tensor(Ground_truth, dtype=torch.float32)  # [N]

    if val_file is None:
        # randomly generate train/val mask
        train_mask, val_mask = random_true_false_pair(Y.shape[0], proportion)
    else:
        # explicitly select val samples
        val_mask = torch.zeros(Y.shape[0], dtype=torch.bool)
        vender_info_df = pd.read_csv(vertex_path)
        vender2index = dict(zip(vender_info_df['vender_id'], vender_info_df['vertex_index']))
        val_vender_df = pd.read_csv(val_file)['vender_id']
        cnt = 0
        for vender_id in val_vender_df:
            index = vender2index[vender_id]
            val_mask[index] = True
            cnt += 1

        print(f'Specify {cnt} instances in the validation set.')
        train_mask = ~val_mask

    result = {
        'X': X,                   # [N, 30]
        'Y': Y,                   # [N]
        'train_mask': train_mask, # [N]
        'val_mask': val_mask      # [N]
    }

    return result


def get_industry_to_vertex_mapping(
    vendor_info_path: str,
    vertex_path: str,
    industry_index_path: str
) -> list:
    vender_info_df = pd.read_csv(vendor_info_path)
    vertex_df = pd.read_csv(vertex_path)
    industry_df = pd.read_csv(industry_index_path)

    vender_to_vertex = dict(zip(vertex_df['vender_id'], vertex_df['vertex_index'])) # vender_id -> index
    industry_to_index = dict(zip(industry_df['content'], industry_df['index'])) # industry -> index

    # use defaultdict(list) to aggregate industry_index -> vertex_index mapping
    industry_vertex_map = defaultdict(list)

    for _, row in vender_info_df.iterrows():
        vender_id = row['vender_id']
        vertex_index = vender_to_vertex[vender_id]

        lv1 = row['industry_lv1']
        lv2 = row['industry_lv2']
        lv3 = row['industry_lv3']

        lv1_index = industry_to_index[lv1]
        lv2_index = industry_to_index[lv1+'*'+lv2]
        lv3_index = industry_to_index[lv1+'*'+lv2+'*'+lv3]
        industry_vertex_map[lv1_index].append(vertex_index)
        industry_vertex_map[lv2_index].append(vertex_index)
        industry_vertex_map[lv3_index].append(vertex_index)

    max_index = max(industry_vertex_map.keys())
    result = [industry_vertex_map[i] for i in range(max_index + 1)]

    return result # a list like [[v_i, v_j, ...], [v_k, v_l, ...]], result[i] refers to the vertex that connects to the hyperedge i.


def get_city_to_vertex_mapping(
    order_info_path: str,
    vertex_path: str,
    city_index_path: str,
    start_date: datetime,
    series_len: int
):
    orders_df = pd.read_csv(order_info_path)
    city_df = pd.read_csv(city_index_path)
    vertex_df = pd.read_csv(vertex_path)

    orders_df['send_tm'] = pd.to_datetime(orders_df['send_tm'])
    end_date = start_date + timedelta(days = series_len - 1)
    mask = (orders_df['send_tm'] >= start_date) & (orders_df['send_tm'] <= end_date)
    filtered_orders = orders_df[mask]

    vender_city_pairs = filtered_orders[['vender_id', 'city']].drop_duplicates()

    city_to_idx = dict(zip(city_df['content'], city_df['index']))

    vendor_to_idx = dict(zip(vertex_df['vender_id'], vertex_df['vertex_index']))

    city_to_vendor = defaultdict(set)

    for _, row in vender_city_pairs.iterrows():
        city = row['city']
        vendor = row['vender_id']

        e_idx = city_to_idx[city]
        v_idx = vendor_to_idx[vendor]
        city_to_vendor[e_idx].add(v_idx)

    max_eidx = max(city_df['index'])
    result = []
    for i in range(max_eidx + 1):
        result.append(sorted(list(city_to_vendor.get(i, []))))

    return result


def gen_adjacency_matrix(A):
    max_col = -1
    for row in A:
        max_col = max(max_col, max(row))

    num_rows = len(A)
    num_cols = max_col + 1

    matrix = torch.zeros((num_rows, num_cols), dtype=torch.float32)

    for i, row in enumerate(A):
        for k in row:
            matrix[i, k] = 1.0

    return matrix


def process_data_pipeline(
    data_root_dir: str,
    info_root_dir: str,
    start_date: datetime,
    series_len: int,
    filter_file: str = None,
    val_file: str = None
):
    os.makedirs("./train_data", exist_ok=True)
    # Extract the order data for the `series_len + 7` days that meet the criteria, and write it to `./train_data/orders.csv`.
    filter_venders_by_active_days(
        data_root_dir=data_root_dir,
        output_file="./train_data/orders.csv",
        start_date=start_date,
        series_len=series_len,
        non_zero_days=20,
        filter_file=filter_file
    )

    # TODO Remove '未知' from the original data
    breakpoint()
 
    filter_venders_by_daily_avg(
        input_csv="./train_data/orders.csv", 
        output_csv="./train_data/orders.csv",
        threshold=200,
        start_date=start_date,
        series_len=series_len
    )

    unique = set()
    df = pd.read_csv("./train_data/orders.csv", usecols=['vender_id'])
    unique.update(df['vender_id'].drop_duplicates())
    unique_vendor_id = pd.DataFrame({'vender_id': sorted(unique)})
    print(f"a total of {len(unique)} unique vendor_ids are found.")

    vender_ids = set(unique_vendor_id['vender_id']) # set of unique vendor_ids
    filtered_chunks = []
    for chunk in pd.read_csv(os.path.join(info_root_dir, "vendor_info.csv"), chunksize=100000):
        filtered_chunk = chunk[chunk['vender_id'].isin(vender_ids)]
        filtered_chunks.append(filtered_chunk)

    # write filtered result into a temporary file `vendor_info_temp.csv`
    pd.concat(filtered_chunks, ignore_index=True).to_csv("./train_data/vendor_info_temp.csv", index=False, encoding='utf-8')

    # only keep vendor_id and 3-level industries
    pd.read_csv("./train_data/vendor_info_temp.csv")[['vender_id', 'industry_lv1', 'industry_lv2', 'industry_lv3']].to_csv("./train_data/vendor_info_temp.csv", index=False)

    # count the number of different 3-level industries
    count_element(input_path="./train_data/vendor_info_temp.csv", task='industry', output_path="./train_data/e_ind.csv")

    # count the number of different cities
    count_element(input_path="./train_data/orders.csv", task='city', output_path="./train_data/e_city.csv")

    # get vertex -> index mapping
    df = (
        pd.read_csv("./train_data/vendor_info_temp.csv")[['vender_id']]
        .reset_index()
        .rename(columns={'index': 'vertex_index'})
    )
    df[['vender_id', 'vertex_index']].to_csv("./train_data/vertex.csv", index=False)

    # get adjacency matrix of the industry hypergraph `\mathcal{G}_i`
    e_ind: list = get_industry_to_vertex_mapping(
        vendor_info_path="./train_data/vendor_info_temp.csv", 
        vertex_path="./train_data/vertex.csv", 
        industry_index_path='./train_data/e_ind.csv'
    )

    # get adjacency matrix of the city hypergraph `\mathcal{G}_s`
    e_city: list = get_city_to_vertex_mapping(
        order_info_path='./train_data/orders.csv', 
        vertex_path='./train_data/vertex.csv', 
        city_index_path='./train_data/e_city.csv', 
        start_date=start_date,
        series_len=series_len
    )
    os.remove("./train_data/vendor_info_temp.csv")

    # get dict: {vender_id -> [[1:series_len] daily order volume，total order volume of the subsequent 7 days]}
    order_volume = extract_order_series(order_file="./train_data/orders.csv", start_date=start_date, series_len=series_len)

    # generate torch.Tensor for training
    dataset: dict = feature_to_tensor(
        vertex_path="./train_data/vertex.csv",
        order_volume=order_volume, 
        proportion=0.8,
        val_file=val_file
    )
    
    # generate v2e_weight matrix for `\mathcal{G}_s` (city hypergraph)
    # dataset['v2e_weight'] has a shape of (|E|, |V|), H[i][j] refers to the vendor_j's order volume through city_i / vendor_j's total order volume
    dataset = generate_v2e_weight(
        vertex_info_path="./train_data/vertex.csv",
        hyperedge_info_path="./train_data/e_city.csv",
        order_info_path="./train_data/orders.csv",
        dataset=dataset, 
        start_date=start_date, 
        e_list=e_city, 
        series_len=series_len)

    dataset['H_industry'] = gen_adjacency_matrix(e_ind).t() # torch.Tensor (|V|, |E|)
    dataset['H_city'] = dataset['v2e_weight'].to_dense().t() # torch.Tensor (|V|, |E|)

    # dataset: key: X, Y, train_mask, val_mask, v2e_weight, industry_hyperedge, city_hyperedge
    with open('./train_data/dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)


if __name__ == "__main__":
    process_data_pipeline(
        data_root_dir="../data/buckets",
        info_root_dir="../data",
        start_date=datetime(2022, 3, 1),
        series_len=30,
        filter_file="./filter/common.csv",
        val_file="./filter/val.csv"
    )