import json
import os
from typing import Any, Iterator, Tuple


def _iter_image_label(part: Any) -> Iterator[Tuple[str, Any]]:
    """
    从一个分割（可能是 dict 包含 'data'、也可能是 dict/list）中迭代 (image_path, gt_label)。
    不会处理 meta；若找不到 gt_label 则返回空字符串。
    """
    # 如果是 {'meta':..., 'data': ...} 结构，取 'data'
    if isinstance(part, dict) and 'data' in part:
        data = part['data']
    else:
        data = part

    # 情况 A: data 是 dict（可能是 patient_id -> patient_entry，或 image_id -> image_dict）
    if isinstance(data, dict):
        for v in data.values():
            # 直接是图片字典
            if isinstance(v, dict) and 'image_path' in v:
                yield v.get('image_path'), v.get('gt_label', '')
                continue
            # patient_entry 风格，查找 left/right
            if isinstance(v, dict):
                for side in ('left', 'right'):
                    lst = v.get(side)
                    if isinstance(lst, list):
                        for img in lst:
                            if isinstance(img, dict) and 'image_path' in img:
                                yield img.get('image_path'), img.get('gt_label', '')
            # 某些情况下 value 本身可能是列表
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and 'image_path' in item:
                        yield item.get('image_path'), item.get('gt_label', '')

    # 情况 B: data 是列表（每项可能是图片字典或 patient dict）
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'image_path' in item:
                yield item.get('image_path'), item.get('gt_label', '')
                continue
            if isinstance(item, dict):
                for side in ('left', 'right'):
                    lst = item.get(side)
                    if isinstance(lst, list):
                        for img in lst:
                            if isinstance(img, dict) and 'image_path' in img:
                                yield img.get('image_path'), img.get('gt_label', '')

    # 其它类型忽略


def json2txt(json_path: str, out_dir: str = None, keys=('train', 'val', 'test')) -> dict:
    """
    将 `json_path` 中的 train/val/test 部分各自导出为文本文件。
    输出文件名为: `basename_train.txt` 等。返回字典 mapping key->output_path（未生成的 key 不在返回中）。
    """
    if out_dir is None:
        out_dir = os.path.dirname(json_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    base = os.path.splitext(os.path.basename(json_path))[0]
    written = {}

    for k in keys:
        if k not in obj:
            continue
        samples = list(_iter_image_label(obj[k]))
        if not samples:
            # 如果没有可写样本，则跳过写文件
            continue
        out_path = os.path.join(out_dir, f"{base}_{k}.txt")
        with open(out_path, 'w', encoding='utf-8') as w:
            for image_path, gt in samples:
                if not image_path:
                    continue
                # 保证 gt 可写为字符串
                w.write(f"{image_path},{gt}\n")
        written[k] = out_path
    return written


def process_dir_json2txt(dir_path: str, skip_contains: str = None) -> dict:
    """
    遍历 `dir_path` 下的所有 `.json`（非递归），对每个调用 `json2txt`。
    如果 `skip_contains` 非空，则跳过文件名包含该子串的 json（例如跳过已是 mini 的文件）。
    返回一个 dict: { json_filename: {key: out_path, ...}, ... }
    """
    results = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith('.json'):
            continue
        if skip_contains and skip_contains in fname:
            continue
        src = os.path.join(dir_path, fname)
        try:
            written = json2txt(src, out_dir=dir_path)
            results[fname] = written
        except Exception as e:
            # 失败时记录错误路径为空字典
            results[fname] = {}
    return results



data_root_dir = 'fundus_depression/ukb_dataset'
img_root_dir = os.path.join(data_root_dir, 'images')
data_split_result_dir = os.path.join(data_root_dir, 'data_splits')

process_dir_json2txt(os.path.join(data_root_dir, 'json_to_process'))