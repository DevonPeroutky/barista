import os
import json
from typing import List

from src.models import RedditPostCaption, CrawlerDataParams, QuestionAnswerEntry


def remove_duplicates(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    seen = set()
    new_data = []
    for obj in data:
        # Create a unique identifier for each object based on the fields
        identifier = (obj['submission_id'], obj['comment_id'], obj['image_url'])
        if identifier not in seen:
            seen.add(identifier)
            new_data.append(obj)

    # Write the new data back to the file
    with open(file_path, 'w') as f:
        json.dump(new_data, f, indent=4)


def check():
    output_path = '/home/devonperoutky/dataset/augmented_roast'

    # Load the Reddit post captions
    reddit_post_captions = []
    with open(f'{output_path}/checkpoints/reddit_post_captions.json', 'r') as json_file:
        reddit_post_captions = json.load(json_file)

    reddit_post_captions = [RedditPostCaption.from_json(json_data=post) for post in reddit_post_captions]
    print(f'Loaded {len(reddit_post_captions)} Reddit post captions')

    # Load the entries
    existing_data = []
    file_path = f'{output_path}/augmented/full_dataset.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)

    # Load existing data as QuestionAnswer objects
    entries: List[QuestionAnswerEntry] = [QuestionAnswerEntry.from_json(json_data=post) for post in existing_data]
    print(f'Loaded {len(entries)} entries')

    # filter out the entries that have already been processed
    ids = set([f'{entry.submission_id}_{entry.comment_id}_{entry.image_url}' for entry in entries])
    submission_ids = set([f'{entry.submission_id}' for entry in entries])
    submission_comment_ids = set([f'{entry.submission_id}_{entry.comment_id}' for entry in entries])

    unprocessed = [post for post in reddit_post_captions if post.submission_id not in [entry.submission_id for entry in entries]]
    print(f'Unprocessed: {len(unprocessed)}')
    unprocessed = [p for p in reddit_post_captions if p not in set(entries)]
    print(f'Unprocessed: {len(unprocessed)}')
    unprocessed = [p for p in reddit_post_captions if f'{p.submission_id}_{p.comment_id}_{p.image_url}' in ids]
    processed = [p for p in reddit_post_captions if f'{p.submission_id}' in submission_ids]
    print(f'Processed submission ids: {len(processed)}')

    processed = [p for p in reddit_post_captions if f'{p.submission_id}_{p.comment_id}' in submission_comment_ids]
    print(f'Processed submission_comment ids: {len(processed)}')

    processed = [p for p in reddit_post_captions if f'{p.submission_id}_{p.comment_id}_{p.image_url}' in ids]
    print(f'Processed submission_comment ids: {len(processed)}')


if __name__ == '__main__':
    # remove_duplicates('/home/devonperoutky/dataset/augmented_roast/checkpoints/reddit_post_captions.json')
    check()
