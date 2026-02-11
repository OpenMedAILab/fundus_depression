

import os
import json
import random
from typing import Any, Dict, Iterable, Tuple, List
import pandas as pd

def _iter_images_from_patient(entry: Any) -> Iterable[Dict]:
    """
    从一个 patient entry 中迭代出图片 dict（包含可能的 'gt_label'、'image_path' 等）。
    支持 patient entry 为 dict（含 'left'/'right' 列表或 'data' dict）或直接为图片列表。
    """
    if isinstance(entry, dict):
        # 优先 left/right
        for side in ('left', 'right'):
            lst = entry.get(side)
            if isinstance(lst, list):
                for img in lst:
                    if isinstance(img, dict):
                        yield img
        # 支持 'data' -> mapping 或图片 dicts
        data = entry.get('data')
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict):
                    yield v
        if isinstance(data, list):
            for itm in data:
                if isinstance(itm, dict):
                    yield itm
        # 有些 patient entry 本身可能就是图片字典
        if 'image_path' in entry and isinstance(entry.get('gt_label', None), (int, float, str)):
            yield entry
    elif isinstance(entry, list):
        for itm in entry:
            if isinstance(itm, dict):
                yield itm

def _is_patient_positive(entry: Any) -> bool:
    """
    判断患者是否为阳性：当任意图片的 gt_label == 1 时返回 True。
    """
    for img in _iter_images_from_patient(entry):
        try:
            if int(img.get('gt_label', 0)) == 1:
                return True
        except Exception:
            # 非整数值当作非正例
            continue
    return False

def _count_patient_and_image_stats(patient_map: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """
    给定 patient_id -> entry 的映射，返回 (patient_pos, patient_neg, image_pos, image_neg)
    """
    patient_pos = 0
    patient_neg = 0
    image_pos = 0
    image_neg = 0
    for entry in patient_map.values():
        pos = _is_patient_positive(entry)
        if pos:
            patient_pos += 1
        else:
            patient_neg += 1
        # 统计图片层面的正负
        imgs_found = False
        for img in _iter_images_from_patient(entry):
            imgs_found = True
            try:
                if int(img.get('gt_label', 0)) == 1:
                    image_pos += 1
                else:
                    image_neg += 1
            except Exception:
                image_neg += 1
        # 若没有找到任何图片，视为 0 张图片（不计入图片数）
        if not imgs_found:
            continue
    return patient_pos, patient_neg, image_pos, image_neg

def _ensure_patient_map_from_section(section: Any) -> Tuple[Dict[str, Any], str]:
    """
    将 section（可能是 dict 包含 'data'、或直接的 dict、或 list）标准化为 patient_map（mapping）。
    返回 (patient_map, original_shape), original_shape in {'data_wrapper','dict','list'}。
    对 list 会生成键为 '__idx_{i}' 的 mapping。
    """
    if isinstance(section, dict) and 'data' in section and isinstance(section['data'], dict):
        return dict(section['data']), 'data_wrapper'
    if isinstance(section, dict):
        # 假定是 patient_id -> entry 的形式
        return dict(section), 'dict'
    if isinstance(section, list):
        mapping = {f'__idx_{i}': v for i, v in enumerate(section)}
        return mapping, 'list'
    # 其它类型返回空映射
    return {}, 'dict'

def _reconstruct_section_from_map(patient_map: Dict[str, Any], original_shape: str) -> Any:
    """
    将 patient_map 还原为原始的 section 结构（data_wrapper -> {'data': map}, dict -> map, list -> list(values)）
    """
    if original_shape == 'data_wrapper':
        return {'data': dict(patient_map)}
    if original_shape == 'dict':
        return dict(patient_map)
    if original_shape == 'list':
        # 保持原顺序由键 __idx_{i} 恢复为列表（按 i 排序）
        items = sorted(patient_map.items(), key=lambda kv: int(kv[0].split('_')[-1]) if kv[0].startswith('__idx_') else 0)
        return [v for _, v in items]
    return dict(patient_map)

def downsample_patients_in_json(json_path: str, out_path: str = None, seed: int = None, n_positive_override: int = None) -> Dict[str, Any]:
    """
    对单个 json 文件的 `train` 部分按患者下采样，使阳性患者数与阴性患者数相同（保留所有该患者下的图片）。
    - json_path: 输入 json 文件路径（`.../xxx.json`）
    - out_path: 若为 None，则在同目录下生成 `basename_downsample_patient.json`
    - seed: 随机种子（可选）
    - n_positive_override: 可选，强制以该值作为正例数量（若提供且小于现有正例数量则会下采样正例）
    返回: 包含 paths 与统计信息的字典
    同目录生成 `_downsample_stats.xlsx` 报表，包含降采样前后患者与图片的正负计数。
    """
    if seed is not None:
        random.seed(seed)

    with open(json_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    train_section = obj.get('train')
    if train_section is None:
        raise ValueError("输入 json 不包含 `train` 部分，无法下采样。")

    patient_map, original_shape = _ensure_patient_map_from_section(train_section)
    # 计算原始统计
    before_patient_pos, before_patient_neg, before_image_pos, before_image_neg = _count_patient_and_image_stats(patient_map)

    # 分离阳性与阴性患者键
    pos_keys = [k for k, v in patient_map.items() if _is_patient_positive(v)]
    neg_keys = [k for k in patient_map.keys() if k not in pos_keys]

    # 目标正例数量 = 当前正例数量（或 override）
    target_pos = len(pos_keys) if n_positive_override is None else int(n_positive_override)
    # 若 override 小于现有正例数量，需要下采样正例；否则若 override 大于现有正例数量，尽量不做扩增（保持现状）
    selected_pos_keys = pos_keys
    if n_positive_override is not None and n_positive_override < len(pos_keys):
        selected_pos_keys = random.sample(pos_keys, n_positive_override)

    # 目标阴例数量与阳例相等
    target_neg = target_pos
    # 如果现有阴例少于目标，则无法补全，只能使用全量阴例
    if len(neg_keys) <= target_neg:
        selected_neg_keys = neg_keys
    else:
        # 从阴例中随机抽取 target_neg 个患者
        selected_neg_keys = random.sample(neg_keys, target_neg)

    # 构造新的 patient_map（包含所有选中的正例与阴例）
    new_patient_map = {}
    for k in selected_pos_keys + selected_neg_keys:
        new_patient_map[k] = patient_map[k]

    # 统计 after
    after_patient_pos, after_patient_neg, after_image_pos, after_image_neg = _count_patient_and_image_stats(new_patient_map)

    # 还原 train section 结构
    new_train_section = _reconstruct_section_from_map(new_patient_map, original_shape)
    # 保持可能的 meta 信息：如果原始 train_section 是 data_wrapper 且包含 meta，则保留 meta
    if isinstance(train_section, dict) and 'data' in train_section:
        new_section = dict(train_section)  # copy
        new_section['data'] = new_train_section['data']
        new_train_section = new_section

    # 构造输出对象（其它部分原样）
    new_obj = dict(obj)
    new_obj['train'] = new_train_section

    # 输出路径
    base_dir = os.path.dirname(json_path) or '.'
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    if out_path is None:
        out_path = os.path.join(base_dir, f"{base_name}_downsample_patient.json")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_obj, f, ensure_ascii=False, indent=2)

    # 保存 excel 报表
    stats = {
        'patient_pos': [before_patient_pos, after_patient_pos],
        'patient_neg': [before_patient_neg, after_patient_neg],
        'image_pos': [before_image_pos, after_image_pos],
        'image_neg': [before_image_neg, after_image_neg],
    }
    df = pd.DataFrame(stats, index=['before', 'after'])
    excel_path = os.path.join(base_dir, f"{base_name}_downsample_stats.xlsx")
    df.to_excel(excel_path)

    return {
        'input_json': json_path,
        'output_json': out_path,
        'excel': excel_path,
        'counts_before': {
            'patient_pos': before_patient_pos, 'patient_neg': before_patient_neg,
            'image_pos': before_image_pos, 'image_neg': before_image_neg
        },
        'counts_after': {
            'patient_pos': after_patient_pos, 'patient_neg': after_patient_neg,
            'image_pos': after_image_pos, 'image_neg': after_image_neg
        }
    }

def process_dir_downsample_patients(dir_path: str, skip_contains: str = None, seed: int = None) -> Dict[str, Any]:
    """
    遍历 `dir_path` 下的所有 .json 文件（非递归），对每个执行患者级下采样并生成输出文件与统计表。
    - skip_contains: 若不为 None，则跳过文件名包含该子串的 json（例如跳过已是 downsample 的文件）
    - seed: 随机种子（传递给每个文件以保证可复现）
    返回: mapping filename -> result dict 或 error 信息
    """
    results = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith('.json'):
            continue
        if skip_contains and skip_contains in fname:
            continue
        src = os.path.join(dir_path, fname)
        try:
            res = downsample_patients_in_json(src, out_path=None, seed=seed)
            results[fname] = res
        except Exception as e:
            results[fname] = {'error': str(e)}
    return results


data_root_dir = 'fundus_depression/ukb_dataset'
img_root_dir = os.path.join(data_root_dir, 'images')
data_split_result_dir = os.path.join(data_root_dir, 'data_splits')

process_dir_downsample_patients(os.path.join(data_root_dir, 'json_to_process'))# python