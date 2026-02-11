import json
import pandas as pd
import numpy as np
import os
from pandas.api import types as ptypes

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, average_precision_score,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)

# ================= 1. 环境与路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
JSON_DIR = DATA_DIR
RESULT_DIR = os.path.join(BASE_DIR, 'results')  # 结果保存目录

# 创建结果目录
os.makedirs(RESULT_DIR, exist_ok=True)


SPLIT_FILES = {
    'new_totdepress': os.path.join(JSON_DIR, 'new_totdepress_data_split.json'),
    'baseline_depression': os.path.join(JSON_DIR, 'baseline_depression_data_split.json'),
    'incident_depression': os.path.join(JSON_DIR, 'incident_depression_data_split.json')
}


# ================= 2. 数据处理函数 =================
def load_data(file_path):
    print(f"正在加载数据: {os.path.basename(file_path)} ...")
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        df = pd.read_csv(file_path)

    # 标签映射 1/0
    for col in ['baseline_depression', 'incident_depression']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map({'是': 1, '否': 0})

    return df


# 寻找数据文件
DATA_FILE = os.path.join(DATA_DIR, '1224_ukb_depression_fundus.xlsx')
df_full = load_data(DATA_FILE)



task_name = 'new_totdepress'
# 1. 读取 Split
json_path = SPLIT_FILES.get(task_name)
with open(json_path, 'r') as f:
    splits = json.load(f)

# 提取 ID
train_ids = [int(i) for i in splits['train']['data'].keys()]
val_ids = [int(i) for i in splits['val']['data'].keys()]
test_ids = [int(i) for i in splits['test']['data'].keys()]

# 2. 划分数据
df_train = df_full[df_full['eid_ckd'].isin(train_ids)].copy()
df_val = df_full[df_full['eid_ckd'].isin(val_ids)].copy()
df_test = df_full[df_full['eid_ckd'].isin(test_ids)].copy()


def _format_count_ratio(count, total, percent_decimals=0):
    """
    返回格式: count/total (XX%)
    percent_decimals: 小数位数，用于 percent 格式化（一般使用 0）
    """
    if total in (0, None) or (isinstance(total, float) and np.isnan(total)):
        return ""
    ratio = count / total if total > 0 else 0.0
    percent = format(ratio, f'.{percent_decimals}%')
    return f"{int(count)}/{int(total)} ({percent})"

def _format_mean_std(s, decimals=1):
    """
    返回格式: mean (std)，保留 decimals 位小数
    若无非空值则返回空字符串
    若 std 为 NaN，则用 0.0 显示
    """
    s_nonnull = s.dropna()
    if s_nonnull.shape[0] == 0:
        return ""
    mean = s_nonnull.mean()
    std = s_nonnull.std()
    if np.isnan(std):
        std = 0.0
    fmt = f"{{mean:.{decimals}f}} ({{std:.{decimals}f}})".format(mean=mean, std=std)
    return fmt

def summarize_dataset(df, fields_config):
    """
    返回一个有序的列表 (heading, value) 表示单个数据集的每一行统计值。
    fields_config: 列配置信息字典，key=字段名, value='numeric'|'categorical'|'label'
    label 表示需要统计阳性/总样本/比例的二分类标签（单行）
    """
    rows = []
    n_total = len(df)
    for col, ctype in fields_config.items():
        if ctype == 'label':
            # 单行: 阳性数/总样本数 (百分比，0 位)
            if col not in df.columns:
                rows.append((col, ""))
                continue
            pos_count = int((df[col] == 1).sum())
            rows.append((col, _format_count_ratio(pos_count, n_total, percent_decimals=0)))
        elif ctype == 'numeric':
            # 单行 mean (std)，1 位小数
            if col not in df.columns:
                rows.append((col, ""))
                continue
            rows.append((col, _format_mean_std(df[col], decimals=1)))
        elif ctype == 'categorical':
            # 标题行（不显示数值）
            rows.append((col, ""))
            if col not in df.columns:
                continue
            total_nonnull = int(df[col].notna().sum())
            # 每个类别： count/total_nonnull (百分比，0 位)
            value_counts = df[col].value_counts(dropna=True)
            for category, cnt in value_counts.items():
                rows.append((f"  {category}", _format_count_ratio(int(cnt), total_nonnull, percent_decimals=0)))
            # 总非空与占比（占全部样本的比例）
            total_pct = format(total_nonnull / n_total if n_total > 0 else 0.0, '.0%')
            # rows.append((f"  total_nonnull", f"{total_nonnull} ({total_pct})"))
        else:
            # 未知类型，跳过
            continue
    return rows

def generate_count_statistics(dfs, output_path=None):
    """
    dfs: dict with keys: 'all_data','training_data','validation_data','test_data' mapping to DataFrame
    output_path: 输出的 excel 文件路径，默认使用 RESULT_DIR/count_statistics.xlsx
    """
    # 字段配置：明确哪些字段是 label / numeric / categorical
    fields_config = {
        'new_totdepress': 'label',
        'baseline_depression': 'label',
        'incident_depression': 'label',
        'gender': 'categorical',
        'baselineage': 'numeric',
        'bmi': 'numeric',
        'ethnic': 'categorical',
        'townsend': 'numeric',
        'smokingc': 'categorical',
        'drinkc': 'categorical',
        'edu': 'categorical',
        'hbp': 'categorical',
        'dmstatus': 'categorical',
        'hyperlipidemia': 'categorical',
    }

    # 为每个数据集生成行列表（heading, value）
    all_rows = {}
    headings_order = []
    for key, df in dfs.items():
        rows = summarize_dataset(df, fields_config)
        all_rows[key] = rows
        if not headings_order:
            headings_order = [h for h, v in rows]

    # 合并为 DataFrame，行索引为 heading（保持顺序）
    result_df = pd.DataFrame(index=headings_order, columns=['heading', 'all_data', 'training_data', 'validation_data', 'test_data'])
    result_df['heading'] = result_df.index

    # helper to map list of tuples to dict
    def rows_to_dict(rows):
        return {h: v for h, v in rows}

    mapped = {k: rows_to_dict(v) for k, v in all_rows.items()}

    # 填充每一列
    mapping_keys = {
        'all_data': 'all_data',
        'training_data': 'training_data',
        'validation_data': 'validation_data',
        'test_data': 'test_data'
    }
    for col_key in mapping_keys:
        col_map = mapped.get(col_key, {})
        # 若某些heading缺失则留空
        result_df[col_key] = [col_map.get(h, "") for h in result_df.index]

    # 输出路径
    if output_path is None:
        # 使用当前文件目录下的 RESULT_DIR（如模块中存在则使用；否则放到 ./results_count）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(base_dir, 'results')
        os.makedirs(default_dir, exist_ok=True)
        output_path = os.path.join(default_dir, 'count_statistics.xlsx')

    # 保存为 excel
    result_df.to_excel(output_path, index=False)
    print(f"统计结果已保存到: {output_path}")
    return result_df

# 用法示例（在同一文件中调用现有变量）
dfs = {
    'all_data': df_full,
    'training_data': df_train,
    'validation_data': df_val,
    'test_data': df_test
}
result = generate_count_statistics(dfs)