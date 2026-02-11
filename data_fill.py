import pandas as pd
import numpy as np
import os

def excel_stats(dataset_name, excel_path):
    # 统计表格分布与缺失数据，完成非数值列的数值转换
    # ===================== 1. 读取表格数据 =====================
    df = pd.read_excel(excel_path)

    # ===================== 2. 提取指定列 =====================
    # 定义需要提取的列（包含baseline_depression用于抑郁初筛编码）
    target_cols = ["eid_ckd", "smokingc", "gender", "baselineage", "bmi",
                   "hyperlipidemia", "hbp", "dmstatus", "baseline_depression"]
    df_target = df[target_cols].copy()  # 复制避免原数据修改
    print("\n提取指定列后的数据前5行：")
    print(df_target.head())

    # ===================== 3. 按规则进行编码转换 =====================
    # 3.1 定义各列的映射字典
    mapping_dict = {
        "smokingc": {"never smoker": 0, "ex-smoker": 1, "current smoker": 2},
        "gender": {"male": 1, "female": 2},
        "baseline_depression": {"否": 0, "是": 1},  # 抑郁初筛：否→0，是→1
        # hyperlipidemia/hbp/dmstatus已为0/1，若有文本需补充映射（如{"否":0,"是":1}）
    }

    # 3.2 批量替换类别变量为数值编码
    for col, mapping in mapping_dict.items():
        if dataset_name == "ukb_dataset":
            df_target[col] = df_target[col].map(mapping)
        # 检查是否有未匹配的异常值（如拼写错误）
        if df_target[col].isnull().any():
            print(f"\n警告：{col}列存在未匹配的异常值，数量：{df_target[col].isnull().sum()}")
            # 填充未匹配值为np.nan（或根据业务处理）
            df_target[col] = df_target[col].fillna(np.nan)
    if dataset_name == "ukb_dataset":
        # BMI列保留两位小数（与mc，sy格式一致）
        df_target["bmi"] = df_target["bmi"].round(2)

        # ===================== 缺失值统计表格 =====================
    print("\n" + "=" * 50)
    print("缺失值统计表格", dataset_name)
    print("=" * 50)
    # 计算缺失数量
    missing_count = df_target.isnull().sum()
    # 计算缺失比例（百分比，保留两位小数）
    missing_ratio = (df_target.isnull().mean() * 100).round(2)
    # 构建缺失值统计DataFrame
    missing_stats = pd.DataFrame({
        "列名": missing_count.index,
        "缺失数量": missing_count.values,
        "缺失比例(%)": missing_ratio.values
    })
    # 重置索引并格式化输出
    missing_stats = missing_stats.reset_index(drop=True)
    print(missing_stats.to_string(index=False))

    # ===================== 4. 数据整理后校验 =====================
    print("\n" + "=" * 50)
    print("编码转换后的数据校验")
    print("=" * 50)
    print("\n编码转换后的数据前10行：")
    print(df_target.head(10))

    print("\n数据基本信息（检查类型和缺失值）：")
    print(df_target.info())

    print("\n各列取值统计（验证编码是否正确）：")
    for col in df_target.columns:
        print(f"\n{col}列取值分布：")
        print(df_target[col].value_counts(dropna=False))

    return df_target  # 返回处理后的DataFrame，用于后续填充


def fill_data(dataset_name, df_target, save_path):
    """
    填充表格中的缺失数据
    :param dataset_name: 数据集名称（用于命名保存文件）
    :param df_target: 经excel_stats处理后的DataFrame
    :param save_path: 填充后数据的保存路径
    :return: 填充后的DataFrame
    """
    if df_target is None:
        print(f"\n【{dataset_name}】无有效数据，跳过填充")
        return None

    df_filled = df_target.copy()
    print("\n" + "=" * 50)
    print(f"开始填充缺失数据 - {dataset_name}")
    print("=" * 50)

    # 定义变量类型（根据业务逻辑划分）
    # 分类变量：用众数填充（出现次数最多的值）
    cat_cols = ["smokingc", "gender", "hyperlipidemia", "hbp", "dmstatus", "baseline_depression"]
    cat_cols = [col for col in cat_cols if col in df_filled.columns]
    # 数值变量：年龄用中位数（抗极端值），BMI用均值（更贴合分布）
    num_cols = {
        "baselineage": "median",  # 中位数
        "bmi": "mean"  # 均值
    }
    num_cols = {k: v for k, v in num_cols.items() if k in df_filled.columns}

    # 1. 填充分类变量：众数
    for col in cat_cols:
        if df_filled[col].notna().any():  # 确保列非全空
            mode_val = df_filled[col].mode()[0]  # 取第一个众数
            df_filled[col] = df_filled[col].fillna(mode_val)
            print(f"\n【{dataset_name}】分类变量{col}列：用众数{mode_val}填充缺失值")
        else:
            print(f"\n【{dataset_name}】分类变量{col}列：全为缺失值，填充为0（默认值）")
            df_filled[col] = df_filled[col].fillna(0)

    # 2. 填充数值变量：均值/中位数
    for col, method in num_cols.items():
        if df_filled[col].notna().any():
            if method == "mean":
                fill_val = df_filled[col].mean().round(2)
            else:
                fill_val = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(fill_val)
            print(f"\n【{dataset_name}】数值变量{col}列：用{method}({fill_val})填充缺失值")
        else:
            print(f"\n【{dataset_name}】数值变量{col}列：全为缺失值，填充为0（默认值）")
            df_filled[col] = df_filled[col].fillna(0)

    # 3. 转换分类变量为整数（编码值应为整数）
    for col in cat_cols:
        df_filled[col] = df_filled[col].astype(int)

    # 4. 验证填充结果
    print("\n" + "=" * 50)
    print(f"填充后缺失值统计 - {dataset_name}")
    print("=" * 50)
    after_missing = df_filled.isnull().sum()
    print("填充后各列缺失数量：")
    print(after_missing)

    # 5. 保存填充后的数据
    os.makedirs(save_path, exist_ok=True)  # 创建保存目录
    save_file = os.path.join(save_path, f"filled_{dataset_name}.xlsx")
    df_filled.to_excel(save_file, index=False)
    print(f"\n【{dataset_name}】填充后的数据已保存至：{save_file}")

    return df_filled


if __name__ == "__main__":
    # 表格路径
    ukb_excel_path = "data/1224_ukb_depression_fundus.xlsx"
    mc_excel_path = "data/data_mc_ukb_style.xlsx"
    sy_excel_path = "data/data_sy_ukb_style.xlsx"
    excel_list = {"ukb_dataset": ukb_excel_path,
                  # "mc_dataset": mc_excel_path,
                  # "sy_dataset": sy_excel_path
                  }

    # 存储各数据集处理后的结果
    processed_data = {}

    # 统计数据分布和缺失情况
    for key, value in excel_list.items():
        print("\n\n", "*" * 50, key, "*" * 50)
        df_processed = excel_stats(key, value)
        processed_data[key] = df_processed

    # 开始填充数据
    save_filled_table_path = "data"
    print("\n\n" + "=" * 80)
    print("开始填充所有数据集的缺失数据")
    print("=" * 80)
    filled_data = {}
    for key, df_processed in processed_data.items():
        print("\n\n", "*" * 50, f"填充 {key}", "*" * 50)
        df_filled = fill_data(key, df_processed, save_filled_table_path)
        filled_data[key] = df_filled