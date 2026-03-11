# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

class RedditUserDataset(Dataset):
    def __init__(self, timeline_file_path, min_weeks=1):
        """
        Args:
            timeline_file_path: Path to user_timelines_*.json
            min_weeks: Filter out users with fewer active weeks than this value
        """
        self.data = []
        self.load_and_process_data(timeline_file_path, min_weeks)

    def load_and_process_data(self, file_path, min_weeks):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        print(f"Loading data from {file_path}...")
        
        for username, posts in raw_data.items():
            # 1. Group by week
            weekly_groups = defaultdict(lambda: {
                'texts': [],
                'image_paths': [],
                'scores': [],
                'num_comments': [],
                'post_count': 0,
                'timestamps': [],
                'post_details': []   # NEW: per-post (hour, weekday, score, utc)
            })

            for post in posts:
                # Parse time
                dt = datetime.fromisoformat(post['created_date'])
                # Use (Year, Week) as Key
                week_key = f"{dt.year}-{dt.isocalendar()[1]}"
                
                group = weekly_groups[week_key]
                
                # Aggregate text (Title + Text)
                full_text = f"{post.get('title', '')} {post.get('text', '')}"
                group['texts'].append(full_text)
                
                # Aggregate images
                if 'local_image_paths' in post:
                    group['image_paths'].extend(post['local_image_paths'])
                
                # Aggregate behavior data
                group['scores'].append(post.get('score', 0))
                group['num_comments'].append(post.get('num_comments', 0))
                group['post_count'] += 1
                group['timestamps'].append(dt.hour) # Record posting hour for activity analysis
                # NEW: store per-post detail for within-week LSTM
                group['post_details'].append({
                    'hour': dt.hour,
                    'weekday': dt.weekday(),    # 0=Mon, 6=Sun
                    'score': post.get('score', 0),
                    'utc': dt.timestamp(),
                })

            # 2. Filter users with insufficient data
            if len(weekly_groups) < min_weeks:
                continue

            # 3. Organize into temporal sequence (sorted by time)
            sorted_weeks = sorted(weekly_groups.keys())
            user_sequence = []

            for week in sorted_weeks:
                week_data = weekly_groups[week]
                
                # Return raw data here, convert to Tensor in collate_fn or model later
                user_sequence.append({
                    'week_id': week,
                    'text_content': " [SEP] ".join(week_data['texts']), # Concatenate text
                    'image_paths': week_data['image_paths'],
                    'behavior_feats': {
                        'avg_score': np.mean(week_data['scores']),
                        'total_comments': np.sum(week_data['num_comments']),
                        'post_count': week_data['post_count'],
                        'active_hours': np.mean(week_data['timestamps']) # Average posting hour
                    },
                    'post_details': week_data['post_details'],  # NEW: list of per-post dicts
                })
            
            self.data.append({
                'username': username,
                'timeline': user_sequence
            })

        print(f"Processed {len(self.data)} users with sufficient history.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Usage example
if __name__ == "__main__":
    # Adjust path for testing if run directly
    # Assuming this script is in MISA/src/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_root, 'enhanced_reddit_data', 'user_timelines_20251123_114152.json')
    
    if os.path.exists(file_path):
        dataset = RedditUserDataset(timeline_file_path=file_path)
        if len(dataset) > 0:
            print(f"User 0 has {len(dataset[0]['timeline'])} weeks of data.")
            print("Week 1 Behavior:", dataset[0]['timeline'][0]['behavior_feats'])
    else:
        print(f"File not found: {file_path}")
