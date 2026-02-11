import json
import random
from collections import defaultdict
from typing import Set, List, Optional, Iterator
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def find_ukb_relpath(abs_path: str) -> str:
    """
    返回以 `fundus_depression` 开头的相对路径（如果在 abs_path 中存在）。
    否则返回原始路径。
    """
    norm = os.path.normpath(abs_path)
    parts = norm.split(os.sep)
    if 'fundus_depression' in parts:
        idx = parts.index('fundus_depression')
        return os.path.join(*parts[idx:])
    return abs_path


def collect_images(root_dir: str):
    """
    遍历 root_dir 下的子目录，收集图片路径与标签（1: low quality, 0: normal）。
    返回列表 of (absolute_path, rel_path_starting_with_fundus_depression, label)
    """
    root = Path(root_dir)
    items = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        subname = sub.name.lower()
        label = 1 if 'dark' in subname else 0
        for p in sub.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                abs_p = str(p.resolve())
                rel_p = find_ukb_relpath(abs_p)
                items.append((abs_p, rel_p, label))
    return items


def split_items(items, ratios=(0.8, 0.1, 0.1), seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    n1 = int(n * ratios[0])
    n2 = int(n * (ratios[0] + ratios[1]))
    return items[:n1], items[n1:n2], items[n2:]


def write_txt(list_items, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for _, rel_p, label in list_items:
            f.write(f"{rel_p},{label}\n")


def makedirs(path):
    os.makedirs(path, exist_ok=True)


def safe_symlink(src, dst):
    """
    创建符号链接 dst -> src。如果 dst 已存在则在文件名后加索引避免冲突。
    src 使用绝对路径以确保链接有效。
    """
    src_abs = os.path.abspath(src)
    base, ext = os.path.splitext(os.path.basename(dst))
    dirpath = os.path.dirname(dst)
    candidate = dst
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dirpath, f"{base}_{i}{ext}")
        i += 1
    try:
        os.symlink(src_abs, candidate)
    except OSError:
        # 在极少数情况下符号链接可能失败（权限等），尝试 copy 退化处理
        try:
            import shutil
            shutil.copy2(src_abs, candidate)
        except Exception:
            pass
    return candidate


def create_visualization_links(split_items_map, dl_root):
    """
    split_items_map: dict with keys 'train','val','test' -> list of (abs, rel, label)
    dl_root: directory to create `train/val/test` subfolders and low_quality/normal inside
    """
    for split_name, items in split_items_map.items():
        for label_name in ('low_quality', 'normal'):
            makedirs(os.path.join(dl_root, split_name, label_name))
        for abs_p, rel_p, label in items:
            label_dir = 'low_quality' if label == 1 else 'normal'
            target_dir = os.path.join(dl_root, split_name, label_dir)
            dst = os.path.join(target_dir, os.path.basename(abs_p))
            safe_symlink(abs_p, dst)

def list_images_diff(
    complete_data_dir: str,
    complete_data_subdirs1: Optional[List[str]],
    partial_data: str,
    partial_data_subdir2: Optional[List[str]],
    out_txt: str,
    exts=IMG_EXTS,
    verbose: bool = True
) -> List[str]:
    """
    列出所有在 `complete_data_dir` 中但不在 `partial_data` 指定子目录列表中的图片（基于 basename 比较）。
    subdirs1/subdirs2: 要遍历的子目录名称列表（相对于各自根目录）。若为 None 或空则遍历全部。
    把每行写为: image_path,0，image_path 以 `fundus_depression` 开头的相对路径��式。
    返回写入的图片相对路径列表。
    verbose=True 时打印详细统计信息并检查重复 basenames。
    """
    def iter_files(root: Path, subdirs: Optional[List[str]]) -> Iterator[Path]:
        if not root.exists():
            return
            yield  # type: ignore
        if not subdirs:
            for p in root.rglob('*'):
                if p.is_file() and p.suffix.lower() in exts:
                    yield p
        else:
            for sd in subdirs:
                subpath = (root / sd).resolve()
                if not subpath.exists():
                    continue
                for p in subpath.rglob('*'):
                    if p.is_file() and p.suffix.lower() in exts:
                        yield p

    def gather_map(root: Path, subdirs: Optional[List[str]]):
        """
        返回 (basename -> [Path,...], per_sub_counts(dict subname->count))
        """
        names_map = defaultdict(list)
        per_sub = {}
        if not subdirs:
            cnt = 0
            for p in iter_files(root, None):
                names_map[p.name].append(p.resolve())
                cnt += 1
            per_sub['(all)'] = cnt
            return names_map, per_sub

        for sd in subdirs:
            subpath = (root / sd).resolve()
            cnt = 0
            if not subpath.exists():
                per_sub[sd] = 0
                continue
            for p in subpath.rglob('*'):
                if p.is_file() and p.suffix.lower() in exts:
                    names_map[p.name].append(p.resolve())
                    cnt += 1
            per_sub[sd] = cnt
        return names_map, per_sub

    d1 = Path(complete_data_dir)
    d2 = Path(partial_data)

    map2, per2 = gather_map(d2, partial_data_subdir2)
    map1, per1 = gather_map(d1, complete_data_subdirs1)

    basenames2 = set(map2.keys())
    results: List[str] = []
    seen: Set[str] = set()

    # 对于 map1 中的每个 basename，如果不在 basenames2，则把其所有路径（转为相对路径）加入结果并去重
    for name, paths in map1.items():
        if name not in basenames2:
            for p in paths:
                rel = find_ukb_relpath(str(p))
                if rel not in seen:
                    seen.add(rel)
                    results.append(rel)

    # verbose 输出统计与重复检查
    if verbose:
        def summarize_map(name: str, m: dict, per_sub: dict):
            total_files = sum(len(v) for v in m.values())
            unique_basenames = len(m)
            duplicate_basenames = [k for k, v in m.items() if len(v) > 1]
            print(f"--- {name} statistics ---")
            print(f"subdirs counts: {per_sub}")
            print(f"total files: {total_files}")
            print(f"unique basenames: {unique_basenames}")
            print(f"basename duplicates count: {len(duplicate_basenames)}")
            if duplicate_basenames:
                sample = duplicate_basenames[:5]
                print(f"duplicate examples (basename -> paths):")
                for b in sample:
                    print(f"  {b} -> {len(m[b])} occurrences, sample path: {m[b][0]}")
            print("")

        summarize_map("complete_data_dir (`" + complete_data_dir + "`)", map1, per1)
        summarize_map("partial_data (`" + partial_data + "`)", map2, per2)

        # 交叉重复（basename 同时出现在两边）
        inter = set(map1.keys()) & set(map2.keys())
        print(f"Cross-basename intersection count: {len(inter)}")
        if inter:
            sample = list(inter)[:10]
            print(f"Cross-basename examples: {sample}")
        print(f"Remaining images (to be written to `{out_txt}`): {len(results)}")
        print("------------------------------")

    # 写入文件，格式为: image_path,0 （image_path 为以 `fundus_depression` 开头的相对路径）
    makedirs(os.path.dirname(out_txt) or ".")
    with open(out_txt, 'w', encoding='utf-8') as f:
        for rel_path in results:
            f.write(f"{rel_path},0\n")

    return results



def main():
    base = 'fundus_depression/ukb_dataset/images-v0'
    items = collect_images(base)
    if not items:
        print(f"No images found under `{base}`.")
        return

    train_items, val_items, test_items = split_items(items, ratios=(0.8, 0.1, 0.1), seed=42)

    # 写 txt 文件到 base 目录，路径以 `fundus_depression` 开头
    write_txt(train_items, os.path.join(base, 'train.txt'))
    write_txt(val_items, os.path.join(base, 'val.txt'))
    write_txt(test_items, os.path.join(base, 'test.txt'))

    # 创建 DL 可视化目录并生成符号链接
    dl_dir = os.path.join(base, 'DL')
    create_visualization_links({'train': train_items, 'val': val_items, 'test': test_items}, dl_dir)

    # 打印统计信息
    def stats(lst):
        total = len(lst)
        low = sum(1 for _a, _b, lb in lst if lb == 1)
        normal = total - low
        return total, low, normal

    t_tot, t_low, t_norm = stats(train_items)
    v_tot, v_low, v_norm = stats(val_items)
    te_tot, te_low, te_norm = stats(test_items)

    print(f"train: total={t_tot}, low={t_low}, normal={t_norm}")
    print(f"val:   total={v_tot}, low={v_low}, normal={v_norm}")
    print(f"test:  total={te_tot}, low={te_low}, normal={te_norm}")
    print(f"TXT files and DL visualization created under `{base}`.")


def make_remaining_test_set_for_dark_img_pred():
    """
    为 dark image quality prediction 创建剩余测试集 txt 文件，
    包含所有在 complete_data_dir 但不在 partial_data 指定子目录中的图片。
    """
    complete_data_dir = 'fundus_depression/ukb_dataset/images'
    complete_data_subdirs1 = ['left', 'right']
    partial_data = 'fundus_depression/ukb_dataset/images-v0'
    partial_data_subdir2 =  ['left', 'right', 'left_dark1328', 'right_dark1598']

    out_txt = 'fundus_depression/ukb_dataset/images/remaining_test_dark_img_pred.txt'

    listed_paths = list_images_diff(
        complete_data_dir,
        complete_data_subdirs1,
        partial_data,
        partial_data_subdir2,
        out_txt
    )

    print(f"Listed {len(listed_paths)} images for remaining test set, written to {out_txt}.")


# ----------------

def build_labeled_list_from_subdirs(data_dir: str, subdirs: Optional[List[str]] = None, exts=IMG_EXTS):
    """
    遍历 `data_dir` 下的 subdirs（若 subdirs 为 None 则遍历全部子目录），
    返回 list of (rel_path_starting_with_fundus_depression, label)
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir `{data_dir}` not found")
    items: List[Tuple[str, int]] = []
    seen = set()
    def process_dir(d: Path):
        subname = d.name.lower()
        label = 1 if 'dark' in subname else 0
        for p in d.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                rel = find_ukb_relpath(str(p.resolve()))
                if rel in seen:
                    continue
                seen.add(rel)
                items.append((rel, label))

    if not subdirs:
        for d in root.iterdir():
            if d.is_dir():
                process_dir(d)
    else:
        for sd in subdirs:
            subpath = (root / sd)
            if not subpath.exists():
                print(f"warning: subdir `{sd}` not found under `{data_dir}`")
                continue
            process_dir(subpath)
    # 基本统计输出
    total = len(items)
    positives = sum(1 for _p, lb in items if lb == 1)
    print(f"[build_labeled_list] `{data_dir}` subdirs={subdirs} -> total={total}, positive={positives}")
    return items


def read_prediction_file(pred_txt: str, thd=0.3):
    """
    读取预测文件，文件每行包含三列: image_path, prob_class0, prob_class1
    支持逗号或空白分隔。根据 prob_class1 > 0.5 判定 label=1，否则 0。
    返回 list of (rel_path, label)
    """
    p = Path(pred_txt)
    if not p.exists():
        raise FileNotFoundError(f"prediction file `{pred_txt}` not found")
    items: List[Tuple[str, int]] = []
    seen = set()
    with p.open('r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            parts = re.split(r'[\s,]+', s)
            if len(parts) < 3:
                print(f"skip malformed line: {s}")
                continue
            img_path_raw = parts[0]
            try:
                prob1 = float(parts[2])
            except Exception:
                # 也许是 image_path,prob0,prob1 但分隔其他符号，尝试最后一列为 prob1
                try:
                    prob1 = float(parts[-1])
                except Exception:
                    print(f"skip unparsable prob line: {s}")
                    continue
            rel = find_ukb_relpath(img_path_raw)
            if rel in seen:
                raise AssertionError(f"duplicate image_path in prediction file: `{rel}`")
            seen.add(rel)
            label = 1 if prob1 > thd else 0
            items.append((rel, label))
    total = len(items)
    positives = sum(1 for _p, lb in items if lb == 1)
    print(f"[read_prediction_file] `{pred_txt}` -> total={total}, positive={positives}")

    return items


# python
import shutil
from pathlib import Path
from typing import List, Tuple

def merge_and_save_labels(list1: List[Tuple[str, int]], list2: List[Tuple[str, int]], out_txt: str):
    """
    合并两份 label 列表，断言两份之间无重复，结果也无重复，保存为 out_txt（每行 image_path,label）。
    返回 merged list。
    """
    s1 = {p for p, _ in list1}
    s2 = {p for p, _ in list2}
    inter = s1 & s2
    assert len(inter) == 0, f"overlap between sets detected, examples: {list(inter)[:5]}"
    merged = list1 + list2
    merged_paths = [p for p, _ in merged]
    assert len(merged_paths) == len(set(merged_paths)), "duplicates detected in merged result"
    makedirs(os.path.dirname(out_txt) or ".")
    with open(out_txt, 'w', encoding='utf-8') as f:
        for p, lb in merged:
            f.write(f"{p},{lb}\n")
    total = len(merged)
    pos = sum(1 for _p, lb in merged if lb == 1)
    print(f"[merge_and_save] wrote `{out_txt}` total={total}, positive={pos}")
    return merged


# python
def copy_positive_images(merged_list: List[Tuple[str, int]],
                         dest_root: str = 'fundus_depression/ukb_dataset/images',
                         use_symlink: bool = False):
    """
    将 merged_list 中 label==1 的图片复制或创建符号链接到 `dest_root` 目录下。
    参数:
      - merged_list: list of (rel_path, label)
      - dest_root: 目标根目录 (e.g. `fundus_depression/ukb_dataset/images`)
      - use_symlink: True 则创建符号链接（默认），False 则拷贝文件
    保留源路径中 `ukb_dataset` 之后、`images*` 之后的子目录结构。
    """
    dest_root_p = Path(dest_root)
    dest_root_p.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for rel, lb in merged_list:
        if lb != 1:
            continue
        src_p = Path(rel)
        # ��先尝试相对路径或直接路径
        if not src_p.exists():
            alt = Path.cwd() / rel
            if alt.exists():
                src_p = alt
            else:
                # 尝试在工作树中按 basename 查找
                candidates = list(Path('.').rglob(src_p.name))
                if candidates:
                    src_p = candidates[0]
                else:
                    print(f"[copy_positive_images] warning: source not found for `{rel}`, skipped")
                    skipped += 1
                    continue

        parts = src_p.parts
        if 'ukb_dataset' in parts:
            idx = parts.index('ukb_dataset')
            # remainder 包含 'images-v0' 或 'images' 等，去掉这一层以后再作为目标子路径
            remainder = list(parts[idx+1:])
            if remainder and remainder[0].startswith('images'):
                remainder = remainder[1:]
        else:
            # 若路径不包�� ukb_dataset，则只使用 basename
            remainder = [src_p.name]

        dest_p = dest_root_p.joinpath(*remainder)
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        # 若目标已存在则创建不冲突的文件名
        base = dest_p.stem
        ext = dest_p.suffix
        candidate = dest_p
        i = 1
        while candidate.exists():
            candidate = dest_p.with_name(f"{base}_{i}{ext}")
            i += 1

        try:
            if use_symlink:
                # safe_symlink 已处理存在冲突的情况并在失败时回退为复制
                safe_symlink(str(src_p), str(candidate))
            else:
                shutil.copy2(str(src_p), str(candidate))
            copied += 1
        except Exception as e:
            print(f"[copy_positive_images] failed to place `{src_p}` -> `{candidate}`: {e}")
            skipped += 1

    print(f"[copy_positive_images] done. placed={copied}, skipped={skipped}, dest=`{dest_root}`")

# 修改后的主合并流程（替换原来的 create_merged_labels_for_dark_prediction）
def create_merged_labels_for_dark_prediction():
    # 第一份：来自 images-v0 的子目录标注
    data_dir = 'fundus_depression/ukb_dataset/images-v0'
    data_subdirs = ['left', 'right', 'left_dark1328', 'right_dark1598']
    labeled_set1 = build_labeled_list_from_subdirs(data_dir, data_subdirs)

    # 第二份：来自 prediction txt 的标签（prob -> label），阈值改为 0.5
    pred_txt = 'fundus_depression/ukb_dataset/images/remaining_test_dark_img_prediction.txt'
    labeled_set2 = read_prediction_file(pred_txt, thd=0.5)

    # 合并并保存，函数现在返回 merged 列表
    out_txt = 'fundus_depression/ukb_dataset/images/merged_image_quality_labels.txt'
    merged = merge_and_save_labels(labeled_set1, labeled_set2, out_txt)

    # 复制所有 label==1 的图像到 `fundus_depression/ukb_dataset/images`
    copy_positive_images(merged, dest_root='fundus_depression/ukb_dataset/images_dark')



# python
def load_label_map(label_txt: str):
    """
    读取 `label_txt`（格式: image_path,label 每行），返回 dict: rel_path -> int(label)
    使用 find_ukb_relpath 规范化键，断言无重复。
    """
    p = Path(label_txt)
    if not p.exists():
        raise FileNotFoundError(f"label file `{label_txt}` not found")
    label_map = {}
    with p.open('r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            # 以逗号分割，允许路径中有空格
            parts = s.split(',')
            if len(parts) < 2:
                print(f"[load_label_map] skip malformed line: {s}")
                continue
            img_raw = parts[0].strip()
            try:
                lb = int(parts[1].strip())
            except Exception:
                print(f"[load_label_map] skip unparsable label line: {s}")
                continue
            key = find_ukb_relpath(img_raw)
            if key in label_map:
                raise AssertionError(f"duplicate entry in label file for `{key}`")
            label_map[key] = 1 if lb == 1 else 0
    return label_map


# python
def handle_dark_images(
    label_txt: str = 'fundus_depression/ukb_dataset/images/merged_image_quality_labels.txt',
    json_path: str = 'fundus_depression/ukb_dataset/data_splits/new_totdepress_data_split.json',
    verbose: bool = True,
    sample_limit: int = 5
):
    label_map = load_label_map(label_txt)

    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file `{json_path}` not found")
    with p.open('r', encoding='utf-8') as f:
        data_all = json.load(f)

    for data_split in ['train', 'val', 'test']:
        data = data_all[data_split]['data']
        assert isinstance(data, dict), f"unexpected data format for split `{data_split}`"

        total_images = 0
        gt1_images = 0
        gt1_labeled_dark = 0
        removed_images = 0
        removed_left = 0
        removed_right = 0
        kept_left = 0
        kept_right = 0

        # 新增：被移除图片中按 gt 统计
        removed_gt1 = 0
        removed_gt0 = 0

        removed_examples = []
        kept_examples = []
        removed_eids_samples = []

        eids_to_delete = []

        for eid, info in list(data.items()):
            left_list = info.get('left', [])
            right_list = info.get('right', [])

            def filter_side(entries, side_name):
                nonlocal total_images, gt1_images, gt1_labeled_dark
                nonlocal removed_images, removed_left, removed_right, kept_left, kept_right
                nonlocal removed_examples, kept_examples
                nonlocal removed_gt1, removed_gt0

                new_entries = []
                for item in entries:
                    img_path = item.get('image_path')
                    gt = item.get('gt_label')
                    if img_path is None or gt not in (0, 1):
                        continue
                    total_images += 1
                    if int(gt) == 1:
                        gt1_images += 1

                    rel = find_ukb_relpath(img_path)
                    is_dark = label_map.get(rel, 0) == 1

                    if is_dark:
                        removed_images += 1
                        if side_name == 'left':
                            removed_left += 1
                        else:
                            removed_right += 1
                        if int(gt) == 1:
                            gt1_labeled_dark += 1
                            removed_gt1 += 1
                        else:
                            removed_gt0 += 1
                        if len(removed_examples) < sample_limit:
                            removed_examples.append((eid, side_name, img_path, int(gt), rel))
                    else:
                        new_entries.append(item)
                        if side_name == 'left':
                            kept_left += 1
                        else:
                            kept_right += 1
                        if len(kept_examples) < sample_limit:
                            kept_examples.append((eid, side_name, img_path, int(gt), rel))
                return new_entries

            info['left'] = filter_side(left_list, 'left')
            info['right'] = filter_side(right_list, 'right')

            if not info['left'] and not info['right']:
                eids_to_delete.append(eid)
                if len(removed_eids_samples) < sample_limit:
                    removed_eids_samples.append(eid)

        for eid in eids_to_delete:
            del data[eid]
        removed_eids = len(eids_to_delete)

        pct = (gt1_labeled_dark / gt1_images * 100) if gt1_images > 0 else 0.0

        print("------------------------------------------------------------")
        print(f"[handle_dark_images] split={data_split}")
        print(f"  total_images={total_images}")
        print(f"  gt=1 images={gt1_images}")
        print(f"  gt=1 and labeled dark={gt1_labeled_dark} ({pct:.2f}%)")
        print(f"  removed_images_total={removed_images} (left={removed_left}, right={removed_right})")
        print(f"    removed by gt: gt=1={removed_gt1}, gt=0={removed_gt0}")
        print(f"  kept_images_total={total_images - removed_images} (left={kept_left}, right={kept_right})")
        print(f"  removed_eids={removed_eids}")
        if verbose:
            print("")
            print(f"  sample removed eids (up to {sample_limit}): {removed_eids_samples}")
            if removed_examples:
                print(f"  sample removed images (up to {sample_limit}):")
                for eid, side, img_path, gt, rel in removed_examples:
                    print(f"    eid={eid} side={side} gt={gt} rel=`{rel}` src=`{img_path}`")
            else:
                print("  (no removed image examples)")

            if kept_examples:
                print(f"  sample kept images (up to {sample_limit}):")
                for eid, side, img_path, gt, rel in kept_examples:
                    print(f"    eid={eid} side={side} gt={gt} rel=`{rel}` src=`{img_path}`")
            else:
                print("  (no kept image examples)")
        print("------------------------------------------------------------")

    out_p = p.with_name(p.stem + '_cleaned' + p.suffix)
    with out_p.open('w', encoding='utf-8') as f:
        json.dump(data_all, f, ensure_ascii=False, indent=2)
    print(f"[handle_dark_images] cleaned json written to `{out_p}`")

if __name__ == '__main__':
    # make_remaining_test_set_for_dark_img_pred()
    # create_merged_labels_for_dark_prediction()

    # handle_dark_images()
    pass