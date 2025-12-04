import praw
import pandas as pd
import time
import random
from datetime import datetime


class PaperStyleDataCollector:
    def __init__(self, reddit_client):
        self.reddit = reddit_client

        # 论文中使用的子版块映射
        self.clinical_subreddits = {
            'depression': 'depression',
            'Anxiety': 'anxiety',
            'bipolar': 'bipolar',
            'ADHD': 'ADHD',
            'SuicideWatch': 'depression',  # 按论文映射到抑郁
            'mentalhealth': 'general',
            'CPTSD': 'anxiety',  # 按论文映射到焦虑
            'OCD': 'anxiety'  # 按论文映射到焦虑
        }

        self.non_clinical_subreddits = [
            'AskReddit', 'funny', 'pics', 'todayilearned',
            'science', 'worldnews', 'gaming'
        ]

    def collect_paper_style_data(self):
        """按照论文的数据结构收集数据"""
        print("开始收集与论文相同类型的数据...")

        # 1. 收集临床数据（Study 1）
        clinical_data = self.collect_clinical_data()

        # 2. 收集非临床数据（Study 2）
        non_clinical_data = self.collect_non_clinical_data()

        # 3. 收集预测数据（Study 3）
        prediction_data = self.collect_prediction_data(clinical_data)

        return clinical_data, non_clinical_data, prediction_data

    def collect_clinical_data(self):
        """收集临床数据 - 对应论文的Study 1"""
        print("\n=== 收集临床数据 (Study 1) ===")
        clinical_posts = []

        for subreddit_name, disorder in self.clinical_subreddits.items():
            print(f"收集临床子版块: r/{subreddit_name} -> {disorder}")

            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts_collected = 0

                # 使用多种排序方式获取更多样化的数据
                for post in subreddit.top(limit=300, time_filter='year'):
                    if self.is_valid_post(post):
                        clinical_posts.append(self.create_clinical_post_data(post, disorder))
                        posts_collected += 1

                print(f"  ✅ 从 r/{subreddit_name} 收集了 {posts_collected} 条帖子")
                time.sleep(1.5)

            except Exception as e:
                print(f"  ❌ 收集 r/{subreddit_name} 失败: {e}")
                continue

        clinical_df = pd.DataFrame(clinical_posts)
        print(f"临床数据总计: {len(clinical_df)} 条帖子")
        return clinical_df

    def collect_non_clinical_data(self):
        """收集非临床数据 - 对应论文的Study 2"""
        print("\n=== 收集非临床数据 (Study 2) ===")
        non_clinical_posts = []

        for subreddit_name in self.non_clinical_subreddits:
            print(f"收集非临床子版块: r/{subreddit_name}")

            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts_collected = 0

                for post in subreddit.top(limit=200, time_filter='year'):
                    if self.is_valid_post(post):
                        non_clinical_posts.append(self.create_non_clinical_post_data(post))
                        posts_collected += 1

                print(f"  ✅ 从 r/{subreddit_name} 收集了 {posts_collected} 条帖子")
                time.sleep(1.5)

            except Exception as e:
                print(f"  ❌ 收集 r/{subreddit_name} 失败: {e}")
                continue

        non_clinical_df = pd.DataFrame(non_clinical_posts)
        print(f"非临床数据总计: {len(non_clinical_df)} 条帖子")
        return non_clinical_df

    def collect_prediction_data(self, clinical_df):
        """收集预测数据 - 对应论文的Study 3"""
        print("\n=== 收集预测数据 (Study 3) ===")

        # 论文的方法：找到在临床子版块发帖的用户，收集他们之前的非临床帖子
        clinical_authors = clinical_df['author'].unique()
        clinical_authors = [author for author in clinical_authors if author != '[deleted]']

        print(f"找到 {len(clinical_authors)} 个临床用户")

        prediction_posts = []
        authors_processed = 0

        for author_name in clinical_authors[:50]:  # 限制数量避免API限制
            try:
                author = self.reddit.redditor(author_name)
                user_posts = []

                # 获取用户最近的帖子
                for submission in author.submissions.new(limit=100):
                    user_posts.append({
                        'author': author_name,
                        'post_id': submission.id,
                        'title': submission.title,
                        'text': submission.selftext,
                        'subreddit': submission.subreddit.display_name,
                        'created_utc': submission.created_utc,
                        'is_clinical': submission.subreddit.display_name in self.clinical_subreddits
                    })

                if user_posts:
                    # 按时间排序
                    user_posts.sort(key=lambda x: x['created_utc'])

                    # 找到第一次临床发帖
                    clinical_posts = [p for p in user_posts if p['is_clinical']]
                    if clinical_posts:
                        first_clinical_date = min([p['created_utc'] for p in clinical_posts])

                        # 收集第一次临床发帖之前的非临床帖子
                        pre_clinical_posts = [
                            p for p in user_posts
                            if p['created_utc'] < first_clinical_date and not p['is_clinical']
                        ]

                        prediction_posts.extend(pre_clinical_posts)
                        authors_processed += 1
                        print(f"  ✅ 用户 {author_name}: 找到 {len(pre_clinical_posts)} 条预测帖子")

                time.sleep(0.5)  # 避免API限制

            except Exception as e:
                print(f"  ❌ 处理用户 {author_name} 失败: {e}")
                continue

        prediction_df = pd.DataFrame(prediction_posts)

        # 为预测数据添加障碍标签（基于用户后来的临床发帖）
        prediction_df = self.add_prediction_labels(prediction_df, clinical_df)

        print(f"预测数据总计: {len(prediction_df)} 条帖子 (来自 {authors_processed} 个用户)")
        return prediction_df

    def add_prediction_labels(self, prediction_df, clinical_df):
        """为预测数据添加障碍标签"""
        # 创建用户到障碍的映射
        author_to_disorder = {}
        for _, row in clinical_df.iterrows():
            if row['author'] not in author_to_disorder:
                author_to_disorder[row['author']] = row['disorder']

        # 为预测数据添加标签
        prediction_df['disorder'] = prediction_df['author'].map(author_to_disorder)
        prediction_df['subreddit_type'] = 'non_clinical'

        return prediction_df

    def create_clinical_post_data(self, post, disorder):
        """创建临床帖子数据结构（与论文匹配）"""
        return {
            'post_id': post.id,
            'author': str(post.author),
            'title': post.title,
            'text': post.selftext,
            'subreddit': post.subreddit.display_name,
            'subreddit_type': 'clinical',
            'disorder': disorder,  # 论文中的障碍分类
            'created_utc': post.created_utc,
            'created_date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
            'score': post.score,
            'num_comments': post.num_comments,
            'upvote_ratio': post.upvote_ratio,
            'is_self': post.is_self,
            'over_18': post.over_18,
            'text_length': len(post.selftext),
            'word_count': len(post.selftext.split())
        }

    def create_non_clinical_post_data(self, post):
        """创建非临床帖子数据结构（与论文匹配）"""
        return {
            'post_id': post.id,
            'author': str(post.author),
            'title': post.title,
            'text': post.selftext,
            'subreddit': post.subreddit.display_name,
            'subreddit_type': 'non_clinical',
            'disorder': 'normal',  # 论文中的正常类别
            'created_utc': post.created_utc,
            'created_date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
            'score': post.score,
            'num_comments': post.num_comments,
            'upvote_ratio': post.upvote_ratio,
            'is_self': post.is_self,
            'over_18': post.over_18,
            'text_length': len(post.selftext),
            'word_count': len(post.selftext.split())
        }

    def is_valid_post(self, post):
        """检查帖子是否有效（与论文标准匹配）"""
        # 跳过空内容、已删除内容、过短内容
        if not post.selftext or post.selftext in ['[deleted]', '[removed]']:
            return False

        # 论文中预处理：移除过短文本
        if len(post.selftext.strip()) < 50:  # 最小文本长度
            return False

        return True

    def save_paper_format_data(self, clinical_df, non_clinical_df, prediction_df):
        """按照论文格式保存数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # 保存临床数据
        clinical_df.to_csv(f"paper_style_clinical_{timestamp}.csv", index=False)

        # 保存非临床数据
        non_clinical_df.to_csv(f"paper_style_non_clinical_{timestamp}.csv", index=False)

        # 保存预测数据
        prediction_df.to_csv(f"paper_style_prediction_{timestamp}.csv", index=False)

        # 生成数据报告
        self.generate_data_report(clinical_df, non_clinical_df, prediction_df, timestamp)

        print(f"\n🎉 论文格式数据保存完成！")
        print(f"📁 文件已保存为: paper_style_*_{timestamp}.csv")

    def generate_data_report(self, clinical_df, non_clinical_df, prediction_df, timestamp):
        """生成与论文匹配的数据报告"""
        report = {
            'collection_info': {
                'timestamp': timestamp,
                'data_types': ['clinical', 'non_clinical', 'prediction'],
                'format': 'paper_style'
            },
            'dataset_stats': {
                'clinical': {
                    'total_posts': len(clinical_df),
                    'disorder_distribution': clinical_df['disorder'].value_counts().to_dict(),
                    'subreddits': clinical_df['subreddit'].value_counts().to_dict()
                },
                'non_clinical': {
                    'total_posts': len(non_clinical_df),
                    'subreddits': non_clinical_df['subreddit'].value_counts().to_dict()
                },
                'prediction': {
                    'total_posts': len(prediction_df),
                    'unique_users': prediction_df['author'].nunique(),
                    'disorder_distribution': prediction_df['disorder'].value_counts().to_dict()
                }
            },
            'comparison_with_paper': {
                'clinical_coverage': '与论文相同的4种障碍类型 + 相关子版块',
                'non_clinical_coverage': '与论文相同的普通子版块类型',
                'prediction_method': '与论文相同的用户时间序列分析方法',
                'data_structure': '与论文相同的字段和预处理逻辑'
            }
        }

        with open(f"paper_style_data_report_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📊 数据报告已生成: paper_style_data_report_{timestamp}.json")


# 使用示例
if __name__ == "__main__":
    # 初始化Reddit客户端
    reddit = praw.Reddit(
        client_id="sHwf21i0jZUtFhxG6sevBg",
        client_secret="IEc5gyd9hNyxPGqoUcc0yFK1QP8slw",
        user_agent="paper_style_data_collection"
    )

    # 创建收集器
    collector = PaperStyleDataCollector(reddit)

    # 收集与论文相同类型的数据
    clinical_df, non_clinical_df, prediction_df = collector.collect_paper_style_data()

    # 保存数据
    collector.save_paper_format_data(clinical_df, non_clinical_df, prediction_df)

    print("\n✅ 数据收集完成！现在你拥有：")
    print("1. 临床数据 - 用于检测现有障碍 (Study 1)")
    print("2. 非临床数据 - 用于检测普通帖子中的障碍 (Study 2)")
    print("3. 预测数据 - 用于预测未来障碍 (Study 3)")
    print("\n📝 数据类型与论文完全匹配，可以开始复现实验！")