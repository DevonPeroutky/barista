from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import PIL
import praw
from PIL import Image
import re

import requests
import shutil
import os
import json

from src.utils import flat_map

'''
--------------------------------------------------
Stable Diffusion training set
--------------------------------------------------
For list of subreddits, grab the top X posts for the year/month/alltime
--> For each post
    --> grab (aka Download) the photo (Filter out those with no photo)
    --> grab top X top-level comments. (Filter out URLS, sort by most comments)
        --> For each comment, grab the text.
        --> Create entry for Image & "Caption" aka the comment
'''


@dataclass
class RedditPostCaption:
    submission_id: str
    comment_id: str
    title: str
    comment: str
    image_width: int
    image_height: int
    image_download_path: str
    image_url: str

    def __eq__(self, other):
        return isinstance(other, RedditPostCaption) and self.submission_id == other.submission_id and self.comment_id == other.comment_id and self.image_url == other.image_url

    def __hash__(self):
        return hash((self.submission_id, self.comment_id, self.image_url))

    def to_json(self, prompt_type="basic"):
        return {
            "id": f"{self.submission_id}_{self.comment_id}",
            "submission_id": self.submission_id,
            "title": self.title,
            "image": self.image_download_path,
            "comment_id": self.comment_id,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWrite an insult about this person." if prompt_type == "basic" else f'<image>\n{self.title}{"" if self.title.endswith(".") or self.title.endswith("!") or self.title.endswith("?") else "."} Write an insult about this person.'
                },
                {
                    "from": "gpt",
                    "value": self.comment
                }
            ]
        }

    def to_csv(self):
        return f'{self.submission_id},{self.comment_id},"{self.comment}",{self.image_width},{self.image_height},{self.image_url},{self.image_download_path}'


class SubRedditCrawler:
    client = None
    seen_ids = set()
    reddit_client = praw.Reddit(
        client_id="D22i28j4wPE90ops_cOsFg",
        client_secret="9g3XJp40WOmqy_cBe_-9ZzEImj-iUw",
        user_agent="my user agent",
    )

    def _print_comment_tree(self, comment, indent=0):
        self.seen_ids.add(comment.id)
        if isinstance(comment, praw.models.MoreComments):
            print("---------- Hit a MoreComments object, skipping")
            return

        print(' ' * indent * 4 + f'({comment.id}) ' + comment.body)

        for reply in comment.replies.list():
            if reply.id not in self.seen_ids:
                self._print_comment_tree(reply, indent + 1)

    def _ingest_subreddit(self, subreddit_name, output_path, num_submissions=None, comments_per_submission=10, time_filter='all') -> List[RedditPostCaption]:
        assert time_filter in ["all", "day", "hour", "month", "week", "year"], f'Invalid time filter: {time_filter}'

        print(f'\n\n-------- Ingesting top {num_submissions} submissions from {subreddit_name} for {time_filter}')

        image_dir = output_path + '/images'
        existing_data = []
        image_captions = []

        # Load the existing data
        file_path = f'{output_path}/augmented/full_dataset.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                existing_data = json.load(json_file)
            print(f'Loaded {len(existing_data)} existing image-caption pairs from {file_path}')

        # Get the list of filenames in the images directory
        existing_images = os.listdir(image_dir)
        print(f'Loaded {len(existing_images)} existing downloaded images from {image_dir}')

        submission_generator = self.reddit_client.subreddit(subreddit_name).top(limit=num_submissions, time_filter=time_filter)
        submissions = [sub for sub in submission_generator]
        print(f'Received {len(submissions)} submissions from subreddit {subreddit_name}')

        for submission in submissions:
            print('--------------------------------------------------')
            print(f'Title: {submission.title} (https://reddit.com{submission.permalink})')
            print(f'Self Text: {submission.selftext}')
            print(f'Image: {submission.url}')
            print('--------------------------------------------------')
            comments = submission.comments
            print(f'\nComments ({submission.num_comments}): \n\n')

            # TODO: if submission.id alread in JSON, skip it.
            image_exists = lambda image_path: Path(image_path).exists()
            existing_submission = any([sub.get('submission_id') == submission.id for sub in existing_data])
            image_paths = [f'{image_dir}/{submission.id}_{key}.jpg' for key in submission.media_metadata.keys()] if hasattr(submission, 'media_metadata') else [f'{image_dir}/{submission.id}.jpg']
            print(f'Expected Images:  {image_paths}')
            all_images_downloaded = all([image_exists(image_path) for image_path in image_paths])

            if existing_submission and all_images_downloaded:
                print(f"Skipping submission {submission.id} as it already exists in the dataset!!!!!!!!!!")
                continue

            images = []

            # if image already downloaded, skip it
            if all([image_exists(image_path) for image_path in image_paths]):
                print(f"All {len(image_paths)} images for {submission.id} already downloaded. Skipping...")

                # Read in all the images as PIL images
                for image_path in image_paths:
                    try:
                        image = Image.open(image_path)
                        images.append({
                            'width': image.width,
                            'height': image.height,
                            'image_path': image_path,
                        })
                    except PIL.UnidentifiedImageError as e:
                        print(f"Error opening image {image_path}, skipping submission")
                        continue
            else:

                if hasattr(submission, 'media_metadata'):
                    print("!!!!!!!!!Submission has media metadata, downloading the full gallery of images")

                    # For each key-value pair in the media_metadata dictionary
                    for key, value in submission.media_metadata.items():
                        # Get the url of the image
                        image_url = value.get('s').get('u')
                        image, width, height, download_path = SubRedditCrawler.download_image(image_url,
                                                                                              f'{image_dir}/{submission.id}_{key}.jpg')
                        if image:
                            images.append({
                                'width': width,
                                'height': height,
                                'image_path': download_path,
                            })
                else:
                    image, width, height, download_path = SubRedditCrawler.download_image(submission.url,
                                                                                          f'{image_dir}/{submission.id}.jpg')
                    if image:
                        images.append({
                            'width': width,
                            'height': height,
                            'image_path': download_path,
                        })

            if not images:
                print("No images found, skipping submission!!!!")
                continue

            # Grap the specified top level comments
            top_level_comments = filter(lambda c: not isinstance(c, praw.models.MoreComments),
                                        comments[:comments_per_submission])

            # Filter out top-level comments that contain a markdown url in the comment body using regex
            top_level_comments = list(filter(lambda c: not re.search(r'\[.*\]\(.*\)', c.body), top_level_comments))

            # Filter out top-level comments that contain a markdown url in the comment body using regex
            top_level_comments = list(filter(lambda c: '[deleted]' not in c.body, top_level_comments))

            # For each comment, if the comment body contains "Edit:" or "edit" then remove everything after that
            for comment in top_level_comments:
                if 'EDIT:' in comment.body:
                    comment.body = comment.body.split('EDIT:')[0]
                if 'Edit:' in comment.body:
                    comment.body = comment.body.split('Edit:')[0]
                elif 'edit:' in comment.body:
                    comment.body = comment.body.split('edit:')[0]

            # Replace newlines with spaces
            for comment in top_level_comments:
                comment.body = comment.body.replace('\n', ' ')

            # Escape double quotes in comment body with preprending a double quote
            for comment in top_level_comments:
                comment.body = comment.body.replace('"', '""').strip()

            print(
                f"Going through {len(top_level_comments)} top level comments and {len(images)} images for submission {submission.id}")

            # Output the top level comments
            for top_level_comment in sorted(top_level_comments, key=lambda c: c.score, reverse=True):
                for image in images:
                    image_captions.append(
                        RedditPostCaption(
                            submission_id=submission.id,
                            comment_id=top_level_comment.id,
                            title=submission.title,
                            comment=top_level_comment.body,
                            image_width=image.get("width"),
                            image_height=image.get("height"),
                            image_download_path=image.get("image_path"),
                            image_url=submission.url
                        )
                    )

        print(f'Ingested {len(image_captions)} image-caption pairs from subreddit {subreddit_name}')
        return image_captions

    def ingest_subreddit_as_conversation_json(self, subreddit_name: str, output_path: str, num_submissions: Optional[int] = None, comments_per_submission: int = 10):
        dataset_types = ["basic", "augmented"]

        # Ensure the output directories exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for dataset_type in dataset_types:
            full_path = output_path + '/' + dataset_type
            if not os.path.exists(full_path):
                os.makedirs(full_path)

        # Ensure the output files exists
        augmented_output_path = os.path.join(output_path, 'augmented', 'full_dataset.json')
        basic_output_path = os.path.join(output_path, 'basic', 'full_dataset.json')
        os.makedirs(os.path.dirname(augmented_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(basic_output_path), exist_ok=True)

        # Set up the image directory
        image_dir = output_path + '/images'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        ingest_subreddit = lambda time_filter: self._ingest_subreddit(subreddit_name, output_path, num_submissions, comments_per_submission, time_filter)
        reddit_post_captions = flat_map(ingest_subreddit, ["all", "day", "hour", "month", "week", "year"])
        print(f'Ingested {len(reddit_post_captions)} image-caption pairs in total from subreddit {subreddit_name}')

        # deduplicate in case we pull multiple of the same submission based on submission_id+comment_id
        reddit_post_captions = list(set(reddit_post_captions))
        print(f'{len(reddit_post_captions)} total new image-caption pairs after de-duplicating')

        for prompt_type in dataset_types:
            full_dataset_path = os.path.join(output_path, prompt_type, 'full_dataset.json')

            # Load the existing data
            if os.path.exists(full_dataset_path):
                with open(full_dataset_path, 'r') as json_file:
                    existing_data = json.load(json_file)
                print(f'Loaded {len(existing_data)} existing image-caption pairs from {full_dataset_path}')

            # Add the new data to the existing data
            print(f'Adding {len(reddit_post_captions)} new image-caption pairs to the existing {len(existing_data)} pairs')
            full_ds = existing_data + [p.to_json(prompt_type=prompt_type) for p in reddit_post_captions]

            # Split image_captions into train, test, and validation sets
            train_percentage, test_percentage, validation_percentage = .8, .1, .1
            assert train_percentage + test_percentage + validation_percentage == 1

            train_size = int(train_percentage * len(full_ds))
            test_size = int(test_percentage * len(full_ds))

            train_split = full_ds[0:train_size]
            test_split = full_ds[train_size:train_size + test_size]
            validation_split = full_ds[train_size + test_size:]

            SubRedditCrawler.save_captions_to_dataset(os.path.join(output_path, prompt_type, 'full_dataset.json'), full_ds, prompt_type)
            SubRedditCrawler.save_captions_to_dataset(os.path.join(output_path, prompt_type, 'train_dataset.json'), train_split, prompt_type)
            SubRedditCrawler.save_captions_to_dataset(os.path.join(output_path, prompt_type, 'test_dataset.json'), test_split, prompt_type)
            SubRedditCrawler.save_captions_to_dataset(os.path.join(output_path, prompt_type, 'validation_dataset.json'), validation_split, prompt_type)

    @staticmethod
    def save_captions_to_dataset(dataset_path, dataset, prompt_type):
        print(f'Saving {len(dataset)} captions for prompt type {prompt_type} to {dataset_path}')

        # Save the new data to the dataset
        with open(dataset_path, 'w') as json_file:
            json.dump(dataset, json_file, indent=4)
            print(f'Saved {len(dataset)} image-caption pairs to {dataset_path}')

    @staticmethod
    def download_image(image_url, download_path):
        print("Downloading image from " + image_url)
        response = requests.get(image_url, stream=True)
        with open(download_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)

        print(f'Downloaded image to {download_path}')

        try:
            # Get height and width of image in pixels
            im = Image.open(download_path)
            width, height = im.size
            return im, width, height, download_path
        except PIL.UnidentifiedImageError as e:
            # delete the file
            os.remove(download_path)
            print("Error opening image, skipping submission")
            return None, None, None, None
