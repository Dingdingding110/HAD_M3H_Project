"""
时序心理健康风险数据采集脚本
核心需求：每个用户必须有跨越 3 个月（≥90 天）、≥4 个不同周 的帖子历史
输出格式与 user_timelines_*.json 完全兼容
"""
import praw
import json
import os
import time
import random
import requests
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# ★ 配置区（按需修改）★
# ============================================================
REDDIT_CLIENT_ID     = "sHwf21i0jZUtFhxG6sevBg"
REDDIT_CLIENT_SECRET = "IEc5gyd9hNyxPGqoUcc0yFK1QP8slw"
REDDIT_USER_AGENT    = "temporal_mental_health_v2"

OUTPUT_DIR   = "temporal_reddit_data"           # 输出目录
IMAGE_DIR    = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON  = os.path.join(OUTPUT_DIR, f"user_timelines_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "collection_progress.json")

# 风险 vs 对照子版块（用于采样种子用户）
RISK_SUBREDDITS = [
    "depression_memes", "anxietymemes", "BPDmemes",
    "CPTSDmemes", "traumacore", "MentalHealthIsland",
    "HealfromYourPast",
]
CONTROL_SUBREDDITS = [
    "art", "EarthPorn", "photography",
    "pics", "itookapicture",
    "NatureIsFuckingLit", "SkyPorn",
]

# 时序过滤要求
MIN_DAYS_SPAN   = 90    # 用户发帖时间跨度至少 90 天
MIN_WEEKS       = 4     # 至少覆盖 4 个不同日历周
MIN_POSTS_TOTAL = 8     # 时间跨度内总帖子数至少 8 篇

# 采集规模
MAX_SEED_POSTS_PER_SUB  = 500   # 每个子版块取 top/new 帖子数（用于找种子用户）
MAX_USERS_PER_CLASS     = 300   # 每个类（risk/control）最多收集用户数
MAX_SUBMISSIONS_PER_USER = 200  # 每个用户最多拉取帖子数

# 图片下载
DOWNLOAD_IMAGES = True          # False 则跳过图片下载，加快采集速度
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

# ============================================================


def init_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"processed_users": [], "collected": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def download_image(url: str, save_dir: str, post_id: str, idx: int) -> str | None:
    """下载单张图片，返回本地路径（失败返回 None）"""
    ext = os.path.splitext(url.split('?')[0])[-1].lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = '.jpg'
    filename = f"{post_id}_{idx}{ext}"
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        return save_path
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200 and len(resp.content) > 1024:
            with open(save_path, 'wb') as f:
                f.write(resp.content)
            return save_path
    except Exception:
        pass
    return None


def get_image_urls(submission) -> list[str]:
    """从提交中提取图片 URL"""
    urls = []
    url = submission.url or ""

    # 直接图片链接
    if any(url.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        urls.append(url)
    # Reddit gallery
    elif hasattr(submission, 'gallery_data') and submission.gallery_data:
        try:
            items = submission.gallery_data.get('items', [])
            for item in items:
                media_id = item.get('media_id')
                media = submission.media_metadata.get(media_id, {})
                if media.get('e') == 'Image':
                    image_url = media.get('s', {}).get('u', '').replace('&amp;', '&')
                    if image_url:
                        urls.append(image_url)
        except Exception:
            pass
    # Reddit preview image
    elif hasattr(submission, 'preview') and submission.preview:
        try:
            images = submission.preview.get('images', [])
            if images:
                src = images[0].get('source', {})
                u = src.get('url', '').replace('&amp;', '&')
                if u:
                    urls.append(u)
        except Exception:
            pass
    return urls


def collect_user_posts(reddit, username: str, user_image_dir: str) -> list[dict]:
    """
    拉取用户帖子历史，返回标准格式的帖子列表。
    每次最多等待 30 秒，超时则返回已收集的内容。
    """
    import threading
    posts = []
    done = threading.Event()

    def _fetch():
        try:
            redditor = reddit.redditor(username)
            for submission in redditor.submissions.new(limit=MAX_SUBMISSIONS_PER_USER):
                if done.is_set():
                    break
                post_id = submission.id
                created_dt = datetime.utcfromtimestamp(submission.created_utc)

                local_image_paths = []
                if DOWNLOAD_IMAGES:
                    image_urls = get_image_urls(submission)
                    for idx, img_url in enumerate(image_urls[:5]):
                        local_path = download_image(img_url, user_image_dir, post_id, idx)
                        if local_path:
                            local_image_paths.append(local_path)

                posts.append({
                    "post_id": post_id,
                    "title": submission.title,
                    "text": submission.selftext or "",
                    "subreddit": submission.subreddit.display_name,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "upvote_ratio": getattr(submission, 'upvote_ratio', 0.0),
                    "created_date": created_dt.isoformat(),
                    "local_image_paths": local_image_paths,
                })
        except Exception as e:
            pass  # 超时或其他错误时直接返回已收集内容

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=30)   # 最多等 30 秒
    done.set()           # 通知线程停止
    return posts

def _collect_user_posts_old(reddit, username: str, user_image_dir: str) -> list[dict]:
    """旧实现，保留备用"""
    posts = []
    try:
        redditor = reddit.redditor(username)
        for submission in redditor.submissions.new(limit=MAX_SUBMISSIONS_PER_USER):
            post_id = submission.id
            created_dt = datetime.utcfromtimestamp(submission.created_utc)

            local_image_paths = []
            if DOWNLOAD_IMAGES:
                image_urls = get_image_urls(submission)
                for idx, img_url in enumerate(image_urls[:5]):  # 每帖最多5张
                    local_path = download_image(img_url, user_image_dir, post_id, idx)
                    if local_path:
                        local_image_paths.append(local_path)

            posts.append({
                "post_id": post_id,
                "title": submission.title,
                "text": submission.selftext or "",
                "subreddit": submission.subreddit.display_name,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "upvote_ratio": getattr(submission, 'upvote_ratio', 0.0),
                "created_date": created_dt.isoformat(),
                "local_image_paths": local_image_paths,
            })
        time.sleep(0.5)
    except Exception as e:
        print(f"    [Warning] 获取用户 {username} 帖子失败: {e}")
    return posts


def passes_temporal_filter(posts: list[dict]) -> dict | None:
    """
    检查用户帖子是否满足时序要求。
    返回统计字典，或 None 表示不满足。
    """
    if len(posts) < MIN_POSTS_TOTAL:
        return None

    dates = []
    for p in posts:
        try:
            dates.append(datetime.fromisoformat(p['created_date']))
        except Exception:
            pass

    if not dates:
        return None

    dates.sort()
    span_days = (dates[-1] - dates[0]).days

    if span_days < MIN_DAYS_SPAN:
        return None

    # 计算不同周数
    weeks = set()
    for dt in dates:
        weeks.add(f"{dt.year}-{dt.isocalendar()[1]}")

    if len(weeks) < MIN_WEEKS:
        return None

    return {
        'span_days': span_days,
        'week_count': len(weeks),
        'total_posts': len(posts),
    }


def collect_seed_usernames(reddit, subreddits: list[str], label: int) -> list[tuple[str, int]]:
    """
    从给定子版块的 top/new 帖子中找种子用户名列表。
    返回 [(username, label), ...] 去重列表。
    """
    users = {}
    for sub_name in subreddits:
        print(f"  扫描子版块 r/{sub_name}...")
        try:
            sub = reddit.subreddit(sub_name)
            count = 0
            for post in sub.top(time_filter='year', limit=MAX_SEED_POSTS_PER_SUB // 2):
                author = str(post.author) if post.author else None
                if author and author not in ('[deleted]', 'None', 'AutoModerator'):
                    users[author] = label
                    count += 1
            for post in sub.new(limit=MAX_SEED_POSTS_PER_SUB // 2):
                author = str(post.author) if post.author else None
                if author and author not in ('[deleted]', 'None', 'AutoModerator'):
                    users[author] = label
                    count += 1
            print(f"    → 找到 {count} 个用户")
            time.sleep(1.0)
        except Exception as e:
            print(f"    [Error] r/{sub_name}: {e}")

    return list(users.items())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    print("=" * 60)
    print("时序心理健康数据采集")
    print(f"要求：每用户 ≥{MIN_DAYS_SPAN} 天跨度 / ≥{MIN_WEEKS} 周 / ≥{MIN_POSTS_TOTAL} 帖")
    print("=" * 60)

    reddit = init_reddit()
    progress = load_progress()
    processed = set(progress['processed_users'])
    collected: dict[str, list] = progress['collected']

    # ---- 统计已收集数量 ----
    risk_collected    = sum(1 for v in collected.values() if v.get('__label__') == 1)
    control_collected = sum(1 for v in collected.values() if v.get('__label__') == 0)
    print(f"[恢复] 已处理用户: {len(processed)}, 已合格: risk={risk_collected}, control={control_collected}")

    # ---- Step 1: 收集种子用户 ----
    print("\n[Step 1] 扫描子版块，收集种子用户名...")
    risk_seeds    = collect_seed_usernames(reddit, RISK_SUBREDDITS, label=1)
    control_seeds = collect_seed_usernames(reddit, CONTROL_SUBREDDITS, label=0)

    # 打乱，避免偏差
    random.shuffle(risk_seeds)
    random.shuffle(control_seeds)

    all_seeds = []
    # 交叉合并，保证两类用户均匀采集
    max_len = max(len(risk_seeds), len(control_seeds))
    for i in range(max_len):
        if i < len(risk_seeds):
            all_seeds.append(risk_seeds[i])
        if i < len(control_seeds):
            all_seeds.append(control_seeds[i])

    print(f"\n种子用户: risk={len(risk_seeds)}, control={len(control_seeds)}")

    # ---- Step 2: 逐用户拉取帖子 ----
    print("\n[Step 2] 逐用户拉取帖子历史（需一段时间）...")

    final_data: dict[str, list] = {}   # username -> [post, ...]
    label_map: dict[str, int]  = {}    # username -> 0 or 1

    # 先把已收集的放进来
    for username, user_info in collected.items():
        if isinstance(user_info, dict):
            label_map[username] = user_info.pop('__label__', -1)
            # 实际帖子在 final_data 里需要从 progress 里单独存
    
    # 用独立文件存帖子内容（避免 progress.json 过大）
    POSTS_FILE = os.path.join(OUTPUT_DIR, "posts_cache.json")
    if os.path.exists(POSTS_FILE):
        with open(POSTS_FILE, 'r', encoding='utf-8') as f:
            final_data = json.load(f)

    for username, label in all_seeds:
        if username in processed:
            continue

        # 检查是否已达到该类的目标数量
        current_risk    = sum(1 for u in label_map if label_map[u] == 1)
        current_control = sum(1 for u in label_map if label_map[u] == 0)

        if label == 1 and current_risk >= MAX_USERS_PER_CLASS:
            continue
        if label == 0 and current_control >= MAX_USERS_PER_CLASS:
            continue
        if current_risk >= MAX_USERS_PER_CLASS and current_control >= MAX_USERS_PER_CLASS:
            break

        print(f"  [{current_risk}r/{current_control}c] 处理用户: {username} (label={label})", end='', flush=True)

        # 创建用户图片目录
        sub_dir = "risk_users" if label == 1 else "control_users"
        user_image_dir = os.path.join(IMAGE_DIR, sub_dir, username)
        os.makedirs(user_image_dir, exist_ok=True)

        posts = collect_user_posts(reddit, username, user_image_dir)

        filter_result = passes_temporal_filter(posts)
        processed.add(username)

        if filter_result:
            final_data[username] = posts
            label_map[username] = label
            print(f" ✅ span={filter_result['span_days']}天, weeks={filter_result['week_count']}, posts={filter_result['total_posts']}")
        else:
            span, week_cnt = 0, 0
            if posts:
                try:
                    dates = [datetime.fromisoformat(p['created_date']) for p in posts]
                    span = (max(dates) - min(dates)).days
                    weeks = set(f"{d.year}-{d.isocalendar()[1]}" for d in dates)
                    week_cnt = len(weeks)
                except:
                    pass
            print(f" ✗ 不满足（span={span}天, weeks={week_cnt}, posts={len(posts)}）")

        # 每处理 20 个用户保存一次进度
        if len(processed) % 20 == 0:
            progress['processed_users'] = list(processed)
            progress['collected'] = {u: {'__label__': label_map[u]} for u in label_map}
            save_progress(progress)
            with open(POSTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False)
            print(f"    [进度保存] risk={current_risk}, control={current_control}")

        time.sleep(random.uniform(0.3, 0.8))

    # ---- Step 3: 保存最终数据 ----
    print(f"\n[Step 3] 保存结果...")
    print(f"  合格用户: risk={sum(1 for u in label_map if label_map[u]==1)}, control={sum(1 for u in label_map if label_map[u]==0)}")

    # 保存 user_timelines JSON（格式与现有数据相同）
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"  时间线数据 → {OUTPUT_JSON}")

    # 保存 user_labels JSON
    labels_path = os.path.join(OUTPUT_DIR, "user_labels.json")
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2)
    print(f"  标签数据   → {labels_path}")

    # 输出统计
    print("\n" + "=" * 60)
    print("采集完成！时序分布统计：")
    from collections import Counter
    week_counts = []
    for username, posts in final_data.items():
        weeks = set()
        for p in posts:
            try:
                dt = datetime.fromisoformat(p['created_date'])
                weeks.add(f"{dt.year}-{dt.isocalendar()[1]}")
            except:
                pass
        week_counts.append(len(weeks))
    
    counter = Counter(week_counts)
    for wc in sorted(counter.keys()):
        print(f"  {wc} 周: {counter[wc]} 位用户")
    if week_counts:
        import numpy as np
        print(f"  平均: {np.mean(week_counts):.1f} 周")
    print("=" * 60)


if __name__ == '__main__':
    main()
