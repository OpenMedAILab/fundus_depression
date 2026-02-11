import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, average_precision_score,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
from xgboost import XGBClassifier

# ================= 1. 环境与路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
JSON_DIR = DATA_DIR
RESULT_DIR = os.path.join(BASE_DIR, 'results') # 结果保存目录

# 创建结果目录
os.makedirs(RESULT_DIR, exist_ok=True)

# 寻找数据文件
DATA_FILE = os.path.join(DATA_DIR, '1224_ukb_depression_fundus.xlsx')

if not os.path.isfile(DATA_FILE):
    print(f"错误: 在 {DATA_DIR} 下没找到数据文件，请检查文件名！")
    exit()

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

def get_preprocessor():
    # 对应文档中使用的特征
    numeric_features = ['townsend', 'bmi', 'baselineage'] 
    categorical_features = ['gender', 'smokingc', 'drinkc', 'ethnic', 'edu', 
                            'hbp', 'dmstatus', 'hyperlipidemia']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


def evaluate_predictions(y, y_prob, task_name='', set_name=""):
    """
    计算指标、打印并保存图表。
    - 仅接受 y 和 y_proba；从概率自动推断 y_pred（阈值 = 0.5）。
    - 若 y_proba 为二维（每类概率），会取第二列作为正类概率。
    """
    y = np.asarray(y)

    if y_prob is None:
        raise ValueError("必须提供 y_proba，用于从概率推断 y_pred 并计算 AUC/PR。")

    # 处理 y_proba 可能是二维（n_samples, n_classes）
    y_proba_arr = np.asarray(y_prob)
    if y_proba_arr.ndim == 2 and y_proba_arr.shape[1] > 1:
        prob_pos = y_proba_arr[:, 1]
    else:
        prob_pos = y_proba_arr.ravel()

    # 从概率推断 y_pred
    threshold = 0.5
    y_pred = (prob_pos >= threshold).astype(int)

    # --- A. 计算数值指标 ---
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y, prob_pos)
    except Exception:
        auc_roc = None
    try:
        auc_pr = average_precision_score(y, prob_pos)
    except Exception:
        auc_pr = None

    cm = confusion_matrix(y, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    print(f"=== [{task_name}] {set_name}集 评估结果 ===")
    print(f"准确率 (Accuracy):    {acc:.4f}")
    print(f"精确率 (Precision):   {prec:.4f}")
    print(f"召回率 (Recall):      {rec:.4f}")
    print(f"特异��� (Specificity): {specificity:.4f}")
    print(f"F1分数 (F1-Score):    {f1:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC:              {auc_roc:.4f}")
    else:
        print("AUC-ROC:              未计算 (计算失败或不适用)")
    if auc_pr is not None:
        print(f"AUC-PR:               {auc_pr:.4f}")
    else:
        print("AUC-PR:               未计算 (计算失败或不适用)")
    print("混淆矩阵:")
    print(cm)
    print("-" * 30)

    # --- B. 绘图并保存 ---
    prefix = f"{task_name}_{set_name}"

    # 混淆矩阵
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {task_name} ({set_name})')
    plt.savefig(os.path.join(RESULT_DIR, f'{prefix}_cm.png'))
    plt.close()

    # ROC 曲线（有概率时绘制）
    if prob_pos is not None:
        plt.figure(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y, prob_pos, name=f"{task_name}")
        plt.title(f'ROC Curve - {task_name} ({set_name})')
        plt.plot([0, 1], [0, 1], "k--")
        plt.savefig(os.path.join(RESULT_DIR, f'{prefix}_roc.png'))
        plt.close()

        # PR 曲线
        plt.figure(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y, prob_pos, name=f"{task_name}")
        plt.title(f'PR Curve - {task_name} ({set_name})')
        plt.savefig(os.path.join(RESULT_DIR, f'{prefix}_pr.png'))
        plt.close()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "confusion_matrix": cm
    }


def evaluate_model(model, X, y, task_name, set_name="test"):
    """
    仅负责对模型做预测，然后把结果交给 evaluate_predictions 处理。
    现在只传递概率结果（model.predict_proba）。
    """
    y_prob = model.predict_proba(X)
    return evaluate_predictions(y, y_prob=y_prob, task_name=task_name, set_name=set_name)


def plot_feature_importance(model, task_name):
    """绘制随机森林的特征重要性"""
    # 获取特征名称
    preprocessor = model.named_steps['preprocessor']
    
    # 获取数值特征名
    num_names = preprocessor.transformers_[0][2]
    # 获取类别特征名 (OneHot编码后的名字)
    cat_names = preprocessor.transformers_[1][1]['encoder'].get_feature_names_out(
        preprocessor.transformers_[1][2])
    
    feature_names = np.r_[num_names, cat_names]
    importances = model.named_steps['classifier'].feature_importances_
    
    # 排序
    indices = np.argsort(importances)[::-1]
    top_n = 15 # 只看前15个重要特征
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {task_name}")
    plt.bar(range(top_n), importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), feature_names[indices][:top_n], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'{task_name}_feature_importance.png'))
    plt.close()
    print(f"已保存特征重要性图表: {task_name}_feature_importance.png")

def get_train_val_test(task_name, df_all):
    """
    从 SPLIT_FILES 对应的 JSON 中读取划分索引，构造并返回
    X_train, y_train, X_val, y_val, X_test, y_test。
    若找不到 JSON 或格式异常，返回 None。
    """
    json_path = SPLIT_FILES.get(task_name)
    if not json_path or not os.path.exists(json_path):
        print(f"跳过任务 {task_name}: 找不到对应的 JSON 文件")
        return None

    with open(json_path, 'r') as f:
        splits = json.load(f)

    train_ids = [int(i) for i in splits['train']['data'].keys()]
    val_ids = [int(i) for i in splits['val']['data'].keys()]
    test_ids = [int(i) for i in splits['test']['data'].keys()]

    df_train = df_all[df_all['eid_ckd'].isin(train_ids)].copy()
    df_val = df_all[df_all['eid_ckd'].isin(val_ids)].copy()
    df_test = df_all[df_all['eid_ckd'].isin(test_ids)].copy()

    drop_cols = ['startdate', 'eid_ckd', 'name', 'depression_date',
                 'incident_depression', 'new_totdepress', 'baseline_depression']

    X_train = df_train.drop(columns=drop_cols, errors='ignore')
    y_train = df_train[task_name]

    X_val = df_val.drop(columns=drop_cols, errors='ignore')
    y_val = df_val[task_name]

    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    y_test = df_test[task_name]

    print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_random_forest_experiment(task_name, df_all):

    print(f"\n{'=' * 40}")
    print(f"正在执行任务: {task_name}")
    print(f"{'=' * 40}")

    # 划分数据
    res = get_train_val_test(task_name, df_all)
    if res is None:
        return
    X_train, y_train, X_val, y_val, X_test, y_test = res

    # 两组超参数配置
    param_sets = [
        {"name": "rf_800", "n_estimators": 800, "max_depth": 10},
        {"name": "rf_200", "n_estimators": 200, "max_depth": 10},
    ]

    candidates = []
    for params in param_sets:
        print(f"\n训练并评估候选: {params['name']} (n_estimators={params['n_estimators']})")
        clf = Pipeline(steps=[
            ('preprocessor', get_preprocessor()),
            ('classifier', RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])

        # 在训练集上训练
        clf.fit(X_train, y_train)

        # 在验证集上评估（evaluate_model 会打印/保存图表并返回指标字典）
        val_metrics = evaluate_model(clf, X_val, y_val, task_name, set_name="val")
        auc_val = val_metrics.get("auc_roc")
        auc_val_score = -np.inf if auc_val is None else float(auc_val)

        candidates.append({
            "name": params["name"],
            "params": params,
            "model": clf,
            "val_metrics": val_metrics,
            "val_auc": auc_val_score
        })

    # 将模型选择过程写入文本文件
    selection_path = os.path.join(RESULT_DIR, f"{task_name}_selection_log.txt")
    with open(selection_path, "w", encoding="utf-8") as f:
        f.write(f"Model selection log for {task_name}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
        for c in candidates:
            line = (f"Candidate: {c['name']}\n"
                    f"  params: {c['params']}\n"
                    f"  val_auc: {c['val_auc']}\n"
                    f"  val_metrics: {c['val_metrics']}\n\n")
            f.write(line)
    print(f"已保存模型选择日志: {selection_path}")

    # 选择验证集上 AUC-ROC 最好的模型
    best = max(candidates, key=lambda c: c["val_auc"])
    print(f"\n已选择最优模型: {best['name']} (val_auc={best['val_auc']:.4f})")

    # 使用最优模型在测试集上评估
    print("\n在测试集上进行最终评估...")
    test_metrics = evaluate_model(best["model"], X_test, y_test, task_name, set_name="test")

    # 保存最终评估指标到 Excel（先把不可直接写入表格的值转为字符串）
    flat_metrics = {}
    for k, v in test_metrics.items():
        if isinstance(v, (np.ndarray, list)):
            flat_metrics[k] = np.array(v).tolist() if isinstance(v, np.ndarray) else v
            # 将数组/列表也转为字符串，方便单元格查看
            flat_metrics[k] = str(flat_metrics[k])
        elif v is None:
            flat_metrics[k] = np.nan
        else:
            try:
                flat_metrics[k] = float(v)
            except Exception:
                flat_metrics[k] = str(v)

    df_metrics = pd.DataFrame([flat_metrics])
    metrics_path = os.path.join(RESULT_DIR, f"{task_name}_test_metrics.xlsx")
    df_metrics.to_excel(metrics_path, index=False)
    print(f"已保存测试集评估指标到 Excel: {metrics_path}")

    # 绘制并保存最优模型的特征重要性
    plot_feature_importance(best["model"], task_name)

    print(f"\n任务 {task_name} 完成：验证集选择模型 `{best['name']}`，测试集 AUC-ROC = {test_metrics.get('auc_roc')}")

    return {
        "method": "RandomForest",
        "task": task_name,
        "metrics": flat_metrics,       # flat_metrics 在函数中已有构造
        "selection_log": selection_path
    }

def run_XGBoost_SMOTE_experiment(task_name, df_all):
    print(f"\n{'=' * 40}")
    print(f"正在执行高级实验 (XGBoost + SMOTE): {task_name}")
    print(f"{'=' * 40}")

    # 划分数据
    res = get_train_val_test(task_name, df_all)
    if res is None:
        return
    X_train, y_train, X_val, y_val, X_test, y_test = res

    # --- 定义Pipeline ---
    # 步骤：预处理 -> SMOTE采样(平衡数据) -> XGBoost分类
    pipeline = ImbPipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            # class_weight 在 XGBoost 中通常用 scale_pos_weight，
            # 但既然用了 SMOTE，这里可以先不设，或者设为 1
        ))
    ])

    # --- 超参数搜索空间 ---
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.7, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.9, 1.0],
    }

    print("正在进行超参数搜索 (RandomizedSearch)...")
    # 3折交叉验证寻找最优参
    search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                n_iter=10, scoring='roc_auc', cv=3, verbose=1, random_state=42)

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"最优参数: {search.best_params_}\n")

    # 评估与绘图
    # evaluate_model(best_model, X_val, y_val, task_name, set_name="val")
    test_metrics = evaluate_model(best_model, X_test, y_test, task_name, set_name="test")

    # 绘制特征重要性
    plot_feature_importance(best_model, task_name)

    # 平展 test_metrics 与返回
    flat_metrics = _flatten_metrics(test_metrics)
    return {
        "method": "XGBoost_SMOTE",
        "task": task_name,
        "metrics": flat_metrics,
        "selection_log": None
    }

# python
def _flatten_metrics(metrics):
    """把 evaluate 返回的复杂值平展为可写入表格的标量/字符串形式"""
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, list)):
            flat[k] = str(np.array(v).tolist())
        elif v is None:
            flat[k] = np.nan
        else:
            try:
                flat[k] = float(v)
            except Exception:
                flat[k] = str(v)
    return flat


def test_classifier():
    df_full = load_data(DATA_FILE)

    tasks = ['new_totdepress', 'baseline_depression']
    all_results = {}

    for task in tasks:
        results = []

        rf_res = run_random_forest_experiment(task, df_full)
        if rf_res is not None:
            results.append(rf_res)

        xgb_res = run_XGBoost_SMOTE_experiment(task, df_full)
        if xgb_res is not None:
            results.append(xgb_res)

        all_results[task] = results

    # 将每个 task 的多方法结果写入同一个 Excel 文件，不同 sheet
    summary_path = os.path.join(RESULT_DIR, 'tasks_summary.xlsx')
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        for task, results in all_results.items():
            if not results:
                continue
            # 构建 DataFrame：每行一个 method
            rows = []
            for r in results:
                row = dict(r['metrics'])  # copy
                row['method'] = r.get('method', '')
                rows.append(row)
            df_sheet = pd.DataFrame(rows)
            # sheet 名称使用 task（注意 Excel sheet 名称长度限制）
            sheet_name = task if len(task) <= 31 else task[:31]
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"已保存所有任务方法汇总到 Excel: {summary_path}")

# ================= 5. 主程序入口 =================
if __name__ == "__main__":
    test_classifier()
