"""
数据集质量诊断脚本
检查：重复特征向量、时间线长度分布、特征统计
"""
import os
import sys
import pickle
import json
import numpy as np
import torch
from collections import Counter

# ---- 路径配置 ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_PATH = os.path.join(BASE_DIR, 'enhanced_reddit_data', 'processed_features.pkl')
LABELS_PATH = os.path.join(BASE_DIR, 'enhanced_reddit_data', 'user_labels.json')

def load_data():
    with open(FEATURES_PATH, 'rb') as f:
        raw_data = pickle.load(f)
    with open(LABELS_PATH, 'r') as f:
        label_map = json.load(f)
    return raw_data, label_map

def check_timeline_lengths(raw_data):
    print("\n" + "="*50)
    print("1. 时间线长度分布")
    print("="*50)
    lengths = [len(u['timeline_features']) for u in raw_data]
    counter = Counter(lengths)
    for l in sorted(counter.keys()):
        print(f"  {l} 周: {counter[l]} 位用户")
    print(f"  平均: {np.mean(lengths):.2f}  最短: {min(lengths)}  最长: {max(lengths)}")

def check_duplicate_features(raw_data, label_map):
    print("\n" + "="*50)
    print("2. 重复特征向量检查（取每个用户第一周的拼接特征）")
    print("="*50)

    feature_hashes = []
    user_info = []

    for u in raw_data:
        username = u['username']
        timeline = u['timeline_features']
        if len(timeline) == 0:
            continue
        week0 = timeline[0]
        # 拼接 text + image + behavior
        vec = np.concatenate([
            week0['text_feat'],
            week0['image_feat'],
            week0['behavior_feat']
        ])
        # 用 bytes hash 来检测完全重复
        h = hash(vec.tobytes())
        feature_hashes.append(h)
        label = label_map.get(username, -1)
        user_info.append((username, label, h))

    hash_counter = Counter(feature_hashes)
    duplicates = {h: cnt for h, cnt in hash_counter.items() if cnt > 1}
    print(f"  总用户数: {len(feature_hashes)}")
    print(f"  唯一特征向量数: {len(hash_counter)}")
    print(f"  重复特征向量组数: {len(duplicates)}")
    total_dup_users = sum(cnt for cnt in duplicates.values())
    print(f"  涉及重复的用户数: {total_dup_users}")

    if duplicates:
        print("\n  前5组重复样本的用户信息:")
        shown = 0
        for h, cnt in sorted(duplicates.items(), key=lambda x: -x[1]):
            if shown >= 5:
                break
            group = [(u, l) for u, l, fh in user_info if fh == h]
            labels_in_group = [l for _, l in group]
            print(f"    组大小={cnt}, 标签分布={Counter(labels_in_group)}, 用户: {[u for u, _ in group[:3]]}{'...' if cnt>3 else ''}")
            shown += 1

def check_feature_stats(raw_data, label_map):
    print("\n" + "="*50)
    print("3. 特征统计（文本/图像/行为）")
    print("="*50)

    text_norms, image_norms, behavior_norms = [], [], []
    image_zero_count = 0
    total_weeks = 0

    for u in raw_data:
        for week in u['timeline_features']:
            t = np.array(week['text_feat'])
            v = np.array(week['image_feat'])
            a = np.array(week['behavior_feat'])
            text_norms.append(np.linalg.norm(t))
            image_norms.append(np.linalg.norm(v))
            behavior_norms.append(np.linalg.norm(a))
            if np.linalg.norm(v) < 1e-6:
                image_zero_count += 1
            total_weeks += 1

    print(f"  总周数样本: {total_weeks}")
    print(f"  文本特征 L2范数: 均值={np.mean(text_norms):.4f}, 最小={np.min(text_norms):.4f}, 零向量数={sum(1 for n in text_norms if n<1e-6)}")
    print(f"  图像特征 L2范数: 均值={np.mean(image_norms):.4f}, 最小={np.min(image_norms):.4f}, 零向量数={image_zero_count} ({100*image_zero_count/total_weeks:.1f}%)")
    print(f"  行为特征 L2范数: 均值={np.mean(behavior_norms):.4f}, 最小={np.min(behavior_norms):.4f}, 零向量数={sum(1 for n in behavior_norms if n<1e-6)}")

def check_full_sequence_duplicates(raw_data, label_map):
    """检查完整时间线是否重复（更严格）"""
    print("\n" + "="*50)
    print("4. 完整时间线重复检查（所有周叠加hash）")
    print("="*50)
    
    seq_hashes = []
    for u in raw_data:
        vecs = []
        for week in u['timeline_features']:
            vecs.append(week['text_feat'])
            vecs.append(week['image_feat'])
            vecs.append(week['behavior_feat'])
        full_vec = np.concatenate(vecs)
        seq_hashes.append(hash(full_vec.tobytes()))

    hash_counter = Counter(seq_hashes)
    duplicates = {h: cnt for h, cnt in hash_counter.items() if cnt > 1}
    print(f"  完全相同时间线的组数: {len(duplicates)}")
    print(f"  涉及用户数: {sum(cnt for cnt in duplicates.values())}")

def check_same_length_feature_distance(raw_data, label_map, max_check=500):
    """对相同长度的用户对，计算特征距离分布"""
    print("\n" + "="*50)
    print("5. 相同长度用户之间的特征相似度采样")
    print("="*50)
    
    # 找长度=1的用户（最容易重复）
    single_week_users = [u for u in raw_data if len(u['timeline_features']) == 1]
    print(f"  只有1周数据的用户数: {len(single_week_users)}")
    
    if len(single_week_users) < 2:
        return
    
    # 抽样计算余弦相似度
    vecs = []
    for u in single_week_users[:50]:
        week = u['timeline_features'][0]
        v = np.concatenate([week['text_feat'], week['image_feat'], week['behavior_feat']])
        vecs.append(v / (np.linalg.norm(v) + 1e-8))
    
    # 计算所有对之间的余弦相似度
    similarities = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            sim = np.dot(vecs[i], vecs[j])
            similarities.append(sim)
    
    if similarities:
        print(f"  采样 {len(vecs)} 个用户, 计算 {len(similarities)} 对相似度")
        print(f"  余弦相似度: 均值={np.mean(similarities):.4f}, 中位数={np.median(similarities):.4f}")
        print(f"  高相似度(>0.95)的对数: {sum(1 for s in similarities if s > 0.95)} ({100*sum(1 for s in similarities if s > 0.95)/len(similarities):.1f}%)")
        print(f"  完全相同(>0.9999)的对数: {sum(1 for s in similarities if s > 0.9999)}")

if __name__ == '__main__':
    print("加载数据...")
    raw_data, label_map = load_data()
    print(f"共 {len(raw_data)} 位用户，标签 {len(label_map)} 条")
    
    check_timeline_lengths(raw_data)
    check_duplicate_features(raw_data, label_map)
    check_feature_stats(raw_data, label_map)
    check_full_sequence_duplicates(raw_data, label_map)
    check_same_length_feature_distance(raw_data, label_map)
    
    print("\n" + "="*50)
    print("诊断完成")
    print("="*50)
