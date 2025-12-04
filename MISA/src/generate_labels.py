import os
import json
import argparse

# Define risk and control subreddits
RISK_SUBREDDITS = {
    'depression_memes', 'anxietymemes', 'BPDmemes', 'CPTSDmemes', 
    'depression_memes', 'HealfromYourPast', 'MentalHealthIsland', 
    'traumacore', 'adhdmeme', 'arttherapy'
}

CONTROL_SUBREDDITS = {
    'art', 'EarthPorn', 'EyeCandy', 'itookapicture', 
    'NatureIsFuckingLit', 'photography', 'pic', 'pics', 
    'SkyPorn', 'wallpapers', 'aww'
}

def generate_labels(data_root, output_path):
    """
    Scans the images directory to assign labels based on subreddit folders.
    Label 1: Risk
    Label 0: Control
    """
    images_root = os.path.join(data_root, 'images')
    if not os.path.exists(images_root):
        print(f"Error: Images directory not found at {images_root}")
        return

    user_labels = {}
    risk_count = 0
    control_count = 0
    
    # Iterate over subreddit folders
    for subreddit in os.listdir(images_root):
        subreddit_path = os.path.join(images_root, subreddit)
        if not os.path.isdir(subreddit_path):
            continue
            
        # Determine label for this subreddit
        label = -1
        if subreddit in RISK_SUBREDDITS:
            label = 1
        elif subreddit in CONTROL_SUBREDDITS:
            label = 0
        else:
            print(f"Skipping unknown subreddit: {subreddit}")
            continue
            
        # Iterate over users in this subreddit
        for username in os.listdir(subreddit_path):
            # Avoid duplicates if a user appears in multiple subreddits (prioritize Risk?)
            # For simplicity, we assume users are unique to subreddits or consistent
            if username not in user_labels:
                user_labels[username] = label
                if label == 1:
                    risk_count += 1
                else:
                    control_count += 1
    
    print(f"Generated labels for {len(user_labels)} users.")
    print(f"Risk (1): {risk_count}")
    print(f"Control (0): {control_count}")
    
    with open(output_path, 'w') as f:
        json.dump(user_labels, f, indent=4)
    print(f"Saved labels to {output_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_root = os.path.join(project_root, 'enhanced_reddit_data')
    output_path = os.path.join(data_root, 'user_labels.json')
    
    generate_labels(data_root, output_path)
