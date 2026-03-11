# -*- coding: utf-8 -*-
"""
update_behavior_features.py
───────────────────────────
Adds rich 16-dim per-week behavior features to existing processed_features.pkl
WITHOUT re-running heavy RoBERTa / ViT models.

Maps raw post timestamps → 16 semantically meaningful weekly features that
capture circadian patterns, posting rhythm, engagement variability, etc.
This is far richer than the original 4-dim stats (avg_score/comments/count/hour).

16-dim feature layout (BEHAVIOR_DIM = 16):
  0  mean_hour_sin   = mean(sin(2pi*hour/24))  -- avg circadian position
  1  mean_hour_cos   = mean(cos(2pi*hour/24))
  2  mean_day_sin    = mean(sin(2pi*weekday/7)) -- avg day-of-week pattern
  3  mean_day_cos    = mean(cos(2pi*weekday/7))
  4  late_night_ratio  = fraction of posts between 22h-5h  (KEY mental health signal)
  5  weekend_ratio     = fraction of posts on Sat/Sun
  6  mean_log_score    = mean(log1p(score))     -- average engagement
  7  std_log_score     = std(log1p(score))      -- engagement variability
  8  mean_log_ivl      = mean(log1p(interval_h))-- avg posting rhythm gap
  9  max_log_ivl       = max(log1p(interval_h)) -- longest silence this week
  10 std_log_ivl       = std(log1p(interval_h)) -- posting rhythm regularity
  11 log_post_count    = log1p(n_posts)          -- activity level
  12 first_hour_sin    = sin(2pi*hour_of_first_post/24) -- onset of activity
  13 first_hour_cos    = cos(2pi*hour_of_first_post/24)
  14 last_hour_sin     = sin(2pi*hour_of_last_post/24)  -- end of activity
  15 last_hour_cos     = cos(2pi*hour_of_last_post/24)
"""

import os
import json
import pickle
import math
import numpy as np
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TIMELINE_DIR = os.path.join(PROJECT_ROOT, 'temporal_reddit_data')
FEATURES_PKL = os.path.join(TIMELINE_DIR, 'processed_features.pkl')

BEHAVIOR_DIM = 16


# ── Helper ────────────────────────────────────────────────────────────────────
def build_week_features(posts_in_week, fallback_hour=12, fallback_score=0):
    """
    Given a list of per-post dicts (hour, weekday, score, utc),
    return a numpy array of shape [BEHAVIOR_DIM=16].

    If posts_in_week is empty, use fallback_hour / fallback_score to build a
    single-post pseudo-entry so the feature vector is never all-zero.
    """
    if not posts_in_week:
        posts_in_week = [{'hour': fallback_hour, 'weekday': 0,
                          'score': fallback_score, 'utc': 0.0}]

    posts_sorted = sorted(posts_in_week, key=lambda p: p['utc'])
    n = len(posts_sorted)

    hours    = np.array([p['hour']    for p in posts_sorted], dtype=np.float32)
    weekdays = np.array([p['weekday'] for p in posts_sorted], dtype=np.float32)
    scores   = np.array([max(p['score'], 0) for p in posts_sorted], dtype=np.float32)

    # Posting intervals in hours
    utcs = np.array([p['utc'] for p in posts_sorted], dtype=np.float64)
    if n > 1:
        ivls = np.diff(utcs) / 3600.0   # seconds → hours
    else:
        ivls = np.array([0.0], dtype=np.float64)

    TWO_PI = 2 * math.pi
    feat = np.zeros(BEHAVIOR_DIM, dtype=np.float32)

    # 0-1: circadian position (mean of all posts)
    feat[0]  = float(np.mean(np.sin(TWO_PI * hours / 24)))
    feat[1]  = float(np.mean(np.cos(TWO_PI * hours / 24)))
    # 2-3: day-of-week pattern
    feat[2]  = float(np.mean(np.sin(TWO_PI * weekdays / 7)))
    feat[3]  = float(np.mean(np.cos(TWO_PI * weekdays / 7)))
    # 4: late-night ratio (22h – 5h)
    feat[4]  = float(np.mean((hours >= 22) | (hours < 5)))
    # 5: weekend ratio
    feat[5]  = float(np.mean(weekdays >= 5))
    # 6-7: engagement stats
    log_sc   = np.log1p(scores)
    feat[6]  = float(np.mean(log_sc))
    feat[7]  = float(np.std(log_sc)) if n > 1 else 0.0
    # 8-10: inter-post interval stats
    log_ivl  = np.log1p(np.clip(ivls, 0, None))
    feat[8]  = float(np.mean(log_ivl))
    feat[9]  = float(np.max(log_ivl))
    feat[10] = float(np.std(log_ivl)) if len(log_ivl) > 1 else 0.0
    # 11: activity level
    feat[11] = float(math.log1p(n))
    # 12-13: onset of activity (first post)
    h0 = float(hours[0])
    feat[12] = math.sin(TWO_PI * h0 / 24)
    feat[13] = math.cos(TWO_PI * h0 / 24)
    # 14-15: end of activity (last post)
    h1 = float(hours[-1])
    feat[14] = math.sin(TWO_PI * h1 / 24)
    feat[15] = math.cos(TWO_PI * h1 / 24)

    return feat   # ndarray(16,)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Find latest timeline JSON
    timeline_files = sorted([
        f for f in os.listdir(TIMELINE_DIR)
        if f.startswith('user_timelines') and f.endswith('.json')
    ])
    if not timeline_files:
        print(f"ERROR: no user_timelines*.json found in {TIMELINE_DIR}")
        return
    timeline_path = os.path.join(TIMELINE_DIR, timeline_files[-1])
    print(f"Using timeline: {timeline_path}")

    # 2. Load raw timeline JSON → build per-user per-week post-detail mapping
    with open(timeline_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} users from JSON.")

    # Build: { username: { week_key: [post_detail, ...] } }
    user_week_posts = {}
    for username, posts in tqdm(raw_data.items(), desc="Parsing timeline"):
        week_map = defaultdict(list)
        for post in posts:
            try:
                dt = datetime.fromisoformat(post['created_date'])
            except Exception:
                continue
            week_key = f"{dt.year}-{dt.isocalendar()[1]}"
            week_map[week_key].append({
                'hour':    dt.hour,
                'weekday': dt.weekday(),
                'score':   post.get('score', 0),
                'utc':     dt.timestamp(),
            })
        user_week_posts[username] = dict(week_map)

    # 3. Load existing pickle
    print(f"Loading existing features from {FEATURES_PKL} ...")
    with open(FEATURES_PKL, 'rb') as f:
        all_data = pickle.load(f)
    print(f"Loaded {len(all_data)} users from pickle.")

    # 4. Add post_seq to each week entry
    missing_users = 0
    total_weeks   = 0
    fallback_weeks = 0

    for user_data in tqdm(all_data, desc="Updating behavior"):
        username = user_data['username']
        week_posts = user_week_posts.get(username, {})
        if not week_posts:
            missing_users += 1

        for week_feat in user_data['timeline_features']:
            week_id = week_feat['week_id']   # e.g. "2024-3"
            posts   = week_posts.get(week_id, [])
            total_weeks += 1

            if posts:
                behavior_16 = build_week_features(posts)
            else:
                # Fallback: use existing aggregated stats as hints
                b = week_feat.get('behavior_feat', [0, 0, 0, 12])
                avg_hour  = int(b[3]) if len(b) > 3 else 12
                avg_score = b[0] if len(b) > 0 else 0
                behavior_16 = build_week_features(
                    [], fallback_hour=avg_hour, fallback_score=avg_score)
                fallback_weeks += 1

            # Save new 16-dim field (keep old behavior_feat for reference)
            week_feat['behavior_feat_rich'] = behavior_16

    print(f"\nDone:")
    print(f"  Total weeks processed: {total_weeks}")
    print(f"  Fallback weeks (no JSON match): {fallback_weeks}")
    print(f"  Users missing from JSON: {missing_users}")

    # 5. Save updated pickle
    with open(FEATURES_PKL, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Saved updated features to {FEATURES_PKL}")

    # Quick sanity check
    sample = all_data[0]['timeline_features'][0]
    print(f"\nSanity check – first user, first week:")
    print(f"  behavior_feat_rich shape: {sample['behavior_feat_rich'].shape}")
    print(f"  feature values: {sample['behavior_feat_rich'].round(3)}")


if __name__ == '__main__':
    main()
