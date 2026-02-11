import json
import os.path
from pathlib import Path
from eval_utils import compute_binary_metrics
from typing import Union, List
import numpy as np


def load_txt_list(txt_path: str,
                  delimiter: str = ',',
                  strip_items: bool = True,
                  convert_numbers: bool = False,
                  skip_blank_lines: bool = True) -> List[List[Union[str, int, float]]]:
    """
    读取 `txt_path`，每行以 `delimiter` 分隔，返回 List[List[...] ]。
    """

    def _try_convert(s: str) -> Union[str, int, float]:
        if s == '':
            return s
        if not convert_numbers:
            return s
        try:
            if '.' in s:
                return float(s)
            return int(s)
        except ValueError:
            return s

    rows = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if skip_blank_lines and line.strip() == '':
                continue
            parts = line.split(delimiter)
            if strip_items:
                parts = [p.strip() for p in parts]
            if convert_numbers:
                parts = [_try_convert(p) for p in parts]
            rows.append(parts)
    return rows



def save_metrics_to_excel(image_result, patient_result, out_path: str = 'image_patient_results.xlsx'):
    import numpy as np
    import pandas as pd

    # 目标列（按用户要求顺序）
    metrics_keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc_roc', 'auc_pr']
    cols = metrics_keys + ['method']

    # 映射：目标列 -> 可能出现在 compute_binary_metrics 返回值中的字段名（按优先级）
    key_aliases = {
        'accuracy': ['accuracy', 'acc'],
        'precision': ['precision'],
        'recall': ['recall', 'recall_sen'],
        'specificity': ['specificity', 'specificity_spe'],
        'f1': ['f1'],
        'auc_roc': ['auc_roc', 'auc'],
        'auc_pr': ['auc_pr', 'ap'],
    }

    def _get_val_from_result(res, desired_key):
        # 尝试从 dict 中按别名取值
        aliases = key_aliases.get(desired_key, [desired_key])
        if isinstance(res, dict):
            for a in aliases:
                if a in res:
                    try:
                        return res[a]
                    except Exception:
                        return np.nan
            return np.nan
        # 如果不是 dict，尝试按属性访问别名
        for a in aliases:
            if hasattr(res, a):
                try:
                    return getattr(res, a)
                except Exception:
                    return np.nan
        return np.nan

    def _row_from_result(res, method_name):
        row = {}
        for k in metrics_keys:
            row[k] = _get_val_from_result(res, k)
            # 强制浮点或 NaN
            try:
                if row[k] is None:
                    row[k] = np.nan
                else:
                    row[k] = float(row[k])
            except Exception:
                row[k] = row[k]  # 保持原样（可能已经是 np.nan）
        row['method'] = method_name
        return row

    rows = [
        _row_from_result(image_result, 'image_level'),
        _row_from_result(patient_result, 'patient_level')
    ]

    df = pd.DataFrame(rows, columns=cols)
    df.to_excel(out_path, index=False)
    print(f"metrics saved to `{out_path}`")


def evaluate_image_level(pred_file: str, gt_file: str) -> dict:
    """
    Image-level evaluation using pred_file and gt_file.

    Args:
        pred_file: Path to prediction file (format: image_path, gt, prob)
        gt_file: Path to ground truth file (format: image_path, label)

    Returns:
        Dictionary containing evaluation metrics
    """
    gt = load_txt_list(os.path.abspath(gt_file), convert_numbers=True)
    pred = load_txt_list(os.path.abspath(pred_file), convert_numbers=True)

    # Build pred mapping: path -> probability (last column is probability)
    pred_map = {}
    for row in pred:
        if not row:
            continue
        path = row[0]
        try:
            prob = float(row[-1])
        except Exception:
            raise ValueError(f"cannot parse probability from pred row: {row}")
        pred_map[path] = prob

    # Align pred with gt order, ensure each gt path has corresponding prediction
    gt_paths = [row[0] for row in gt]
    missing = [p for p in gt_paths if p not in pred_map]
    if missing:
        raise KeyError(f"missing predictions for {len(missing)} images, example: {missing[:5]}")

    pred_aligned = [(p, pred_map[p]) for p in gt_paths]

    for x, y in zip(gt, pred_aligned):
        if x[0] != y[0]:
            raise ValueError(f"mismatched image paths:\ngt: {x[0]}\npred: {y[0]}")

    pred_array = np.array([x[1] for x in pred_aligned])
    gt_array = np.array([x[1] for x in gt])
    image_result = compute_binary_metrics(y_true=gt_array, y_prob=pred_array)
    print(f"image level result: {image_result}")
    return image_result


def evaluate_image_and_patient_level(pred_file: str, json_file: str, verbose: bool = True) -> tuple:
    """
    Both image-level and patient-level evaluation using pred_file and json_file.

    Args:
        pred_file: Path to prediction file (format: image_path, [any], prob)
        json_file: Path to JSON file containing patient-level data structure (authoritative gt source)
        verbose: Whether to print results during evaluation

    Returns:
        Tuple of (image_result, patient_result) dictionaries
    """
    # Load JSON first to get authoritative ground truth labels
    p = Path(json_file)
    if not p.exists():
        raise FileNotFoundError(f"JSON file `{json_file}` not found")
    with p.open('r', encoding='utf-8') as f:
        data_all = json.load(f)

    # Build gt mapping from JSON: image_path -> gt_label
    gt_map = {}
    test_data = data_all['test']['data']
    for eid, info in list(test_data.items()):
        for side in ['left', 'right']:
            for item in info.get(side, []):
                img_path = item.get('image_path')
                gt = item.get('gt_label')
                if img_path:
                    gt_map[img_path] = int(gt)

    # Load predictions
    pred = load_txt_list(os.path.abspath(pred_file), convert_numbers=True)

    # Build image-level prediction map
    image_level_pred = {}
    for row in pred:
        if not row or len(row) < 2:
            continue
        img_path = row[0]
        # Skip header rows (non-numeric values)
        # prob is in the last column
        try:
            prob = float(row[-1])
        except (ValueError, TypeError):
            continue
        image_level_pred[img_path] = prob

    # Compute image-level metrics using gt from JSON
    image_preds = []
    image_gts = []
    for img_path, prob in image_level_pred.items():
        if img_path in gt_map:
            image_preds.append(prob)
            image_gts.append(gt_map[img_path])

    image_result = compute_binary_metrics(y_true=np.array(image_gts), y_prob=np.array(image_preds))
    if verbose:
        print(f"image level result: {image_result}")

    # Compute patient-level metrics
    patient_level_gt = []
    patient_level_pred = []
    for eid, info in list(test_data.items()):
        patient_pred = []
        patient_gt = []
        for side in ['left', 'right']:
            for item in info.get(side, []):
                img_path = item.get('image_path')
                gt = item.get('gt_label')
                if not img_path:
                    continue
                if img_path not in image_level_pred:
                    raise KeyError(f"missing prediction for image `{img_path}`")
                prob = image_level_pred[img_path]
                patient_pred.append(float(prob))
                patient_gt.append(int(gt))

        if not patient_pred:
            continue

        avg_score = sum(patient_pred) / len(patient_pred)
        unique_gts = set(patient_gt)
        assert len(unique_gts) == 1, f"multiple gt values for patient `{eid}`: {unique_gts}"

        patient_level_pred.append(avg_score)
        patient_level_gt.append(next(iter(unique_gts)))

    patient_result = compute_binary_metrics(y_true=np.array(patient_level_gt), y_prob=np.array(patient_level_pred))
    if verbose:
        print(f"patient level result: {patient_result}")

    return image_result, patient_result


# ============== Independent test functions for specific experiments ==============

def test_retfound_newtodress_20epoch():
    """Test function for retfound-newtodress-20epoch model."""
    pred_file = 'fundus_depression/UKB_Project/data/retfound-newtodress-20epoch-pred.txt'
    json_file = 'fundus_depression/ukb_dataset/data_splits/new_totdepress_data_split.json'
    # gt_file for image-level only: 'fundus_depression/ukb_dataset/data_splits/txt/new_totdepress_data_split_test.txt'

    image_result, patient_result = evaluate_image_and_patient_level(pred_file, json_file)
    save_metrics_to_excel(
        image_result, patient_result,
        out_path='fundus_depression/UKB_Project/results/important/'
                 'retfound-newtodress-20epoch-pred_image_patient_results.xlsx'
    )
    return image_result, patient_result


def test_retfound_newtodress_filtered_dark_image_20epoch():
    """Test function for retfound-newtodress with filtered dark images, 20 epochs."""
    pred_file = 'fundus_depression/UKB_Project/data/retfound-newtodress-filtered_dark_image-20epoch-pred.txt'
    json_file = 'fundus_depression/ukb_dataset/data_splits/new_totdepress_data_split_cleaned.json'

    image_result, patient_result = evaluate_image_and_patient_level(pred_file, json_file)
    save_metrics_to_excel(
        image_result, patient_result,
        out_path='fundus_depression/UKB_Project/results/important/'
                 'retfound-newtodress-cleaned-20epoch-pred_image_patient_results.xlsx'
    )
    return image_result, patient_result


def batch_evaluate_all_txt_files(
    input_dir: str = 'depression_dataset/fundus_depression/UKB_Project/data/retfound_visionFM',
    json_file: str = 'fundus_depression/ukb_dataset/data_splits/new_totdepress_data_split.json',
    out_path: str = None
):
    """
    Iteratively run image and patient level evaluation for all txt files in input_dir.
    Save all results to a single Excel file.

    Args:
        input_dir: Directory containing prediction txt files
        json_file: Path to JSON file for patient-level data structure
        out_path: Output Excel file path. If None, defaults to input_dir/batch_results.xlsx
    """
    import pandas as pd
    import glob

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory `{input_dir}` not found")

    txt_files = sorted(glob.glob(str(input_path / '*.txt')))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return None

    print("=" * 80)
    print(f"Batch Evaluation - Found {len(txt_files)} txt files")
    print(f"Input directory: {input_dir}")
    print("=" * 80)

    metrics_keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc_roc', 'auc_pr']
    metrics_short = ['acc', 'prec', 'recall', 'spec', 'f1', 'auc', 'ap']
    key_aliases = {
        'accuracy': ['accuracy', 'acc'],
        'precision': ['precision'],
        'recall': ['recall', 'recall_sen'],
        'specificity': ['specificity', 'specificity_spe'],
        'f1': ['f1'],
        'auc_roc': ['auc_roc', 'auc'],
        'auc_pr': ['auc_pr', 'ap'],
    }

    def _get_val(res, key):
        aliases = key_aliases.get(key, [key])
        if isinstance(res, dict):
            for a in aliases:
                if a in res:
                    try:
                        return float(res[a])
                    except (ValueError, TypeError):
                        return np.nan
        return np.nan

    def _print_result_row(level, result):
        vals = [_get_val(result, k) for k in metrics_keys]
        val_strs = [f"{v:.4f}" if not np.isnan(v) else "  N/A " for v in vals]
        print(f"  {level:8s} | " + " | ".join(val_strs))

    image_rows = []
    patient_rows = []
    for i, txt_file in enumerate(txt_files, 1):
        file_name = Path(txt_file).stem
        print(f"\n[{i}/{len(txt_files)}] {file_name}")
        print("-" * 70)
        header = "  {:8s} | ".format("Level") + " | ".join([f"{s:>6s}" for s in metrics_short])
        print(header)
        print("-" * 70)

        try:
            image_result, patient_result = evaluate_image_and_patient_level(txt_file, json_file, verbose=False)

            _print_result_row("Image", image_result)
            _print_result_row("Patient", patient_result)

            # Image-level row
            img_row = {'file': file_name}
            for k in metrics_keys:
                img_row[k] = _get_val(image_result, k)
            image_rows.append(img_row)

            # Patient-level row
            pat_row = {'file': file_name}
            for k in metrics_keys:
                pat_row[k] = _get_val(patient_result, k)
            patient_rows.append(pat_row)

        except Exception as e:
            print(f"  ERROR: {e}")
            # Add error rows
            img_row = {'file': file_name}
            pat_row = {'file': file_name}
            for k in metrics_keys:
                img_row[k] = np.nan
                pat_row[k] = np.nan
            image_rows.append(img_row)
            patient_rows.append(pat_row)

    # Create DataFrames
    cols = ['file'] + metrics_keys
    df_image = pd.DataFrame(image_rows, columns=cols)
    df_patient = pd.DataFrame(patient_rows, columns=cols)

    # Helper function to print table
    def _print_summary_table(title, rows):
        print("\n" + "=" * 90)
        print(f"{title}")
        print("=" * 90)
        file_width = max(25, max(len(row['file']) for row in rows) + 2)
        header = f"{'File':<{file_width}} | " + " | ".join([f"{s:>6s}" for s in metrics_short])
        print(header)
        print("-" * len(header))
        for row in rows:
            vals = [row[k] for k in metrics_keys]
            val_strs = [f"{v:>6.4f}" if not np.isnan(v) else "   N/A" for v in vals]
            print(f"{row['file']:<{file_width}} | " + " | ".join(val_strs))
        print("=" * 90)

    # Print final summary tables
    _print_summary_table("IMAGE LEVEL RESULTS", image_rows)
    _print_summary_table("PATIENT LEVEL RESULTS", patient_rows)

    # Save to Excel with two sheets
    if out_path is None:
        out_path = str(input_path / 'batch_results.xlsx')

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df_image.to_excel(writer, sheet_name='image_level', index=False)
        df_patient.to_excel(writer, sheet_name='patient_level', index=False)

    print(f"\nResults saved to: {out_path} (2 sheets: image_level, patient_level)")
    return df_image, df_patient


def main():
    # Example 1: Image-level only evaluation
    # image_result = evaluate_image_level(
    #     pred_file='path/to/pred.txt',
    #     gt_file='path/to/gt.txt'
    # )

    # Example 2: Both image and patient level evaluation
    # image_result, patient_result = evaluate_image_and_patient_level(
    #     pred_file='path/to/pred.txt',
    #     json_file='path/to/data.json'
    # )

    # Example 3: Batch evaluate all txt files in a directory
    # batch_evaluate_all_txt_files(
    #     input_dir='UKB_Project/data/retfound_visionFM',
    #     json_file='ukb_dataset/data_splits/new_totdepress_data_split.json',
    #     out_path='UKB_Project/data/retfound_visionFM/batch_results.xlsx'
    # )

    # Run specific test
    # test_retfound_newtodress_filtered_dark_image_20epoch()
    batch_evaluate_all_txt_files(
        input_dir='UKB_Project/data/mmpretrain',
        json_file='ukb_dataset/data_splits/new_totdepress_data_split.json',
        out_path='UKB_Project/data/mmpretrain/batch_results.xlsx'
    )

if __name__ == "__main__":
    main()



