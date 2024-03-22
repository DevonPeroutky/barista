import time
from pathlib import Path
from typing import List, Optional, TypeVar, Generic, Dict

import PIL
import praw
from PIL import Image
import re

import requests
import shutil
import os
import json

from prawcore import TooManyRequests

from src.utils import flat_map
from models import RedditPostCaption, CrawlerDataParams


class SubmissionAlreadyDownloadedException(Exception):
    def __init__(self, message="Submission already exists"):
        self.message = message
        super().__init__(self.message)


class SubRedditCrawler:
    client = None
    seen_ids = set()
    reddit_client = praw.Reddit(
        client_id="D22i28j4wPE90ops_cOsFg",
        client_secret="9g3XJp40WOmqy_cBe_-9ZzEImj-iUw",
        user_agent="my user agent",
    )
    crawler_data_params = None

    def __init__(self, params: CrawlerDataParams):
        self.crawler_data_params = params

    def _print_comment_tree(self, comment, indent=0):
        self.seen_ids.add(comment.id)
        if isinstance(comment, praw.models.MoreComments):
            print("---------- Hit a MoreComments object, skipping")
            return

        print(' ' * indent * 4 + f'({comment.id}) ' + comment.body)

        for reply in comment.replies.list():
            if reply.id not in self.seen_ids:
                self._print_comment_tree(reply, indent + 1)

    # TODO:
    # 1. Make image downloading optional
    # 2. Optional transformation of the title + selftext --> question?
    def _ingest_subreddit(self, subreddit_name, output_path, num_submissions: int = None, comments_per_submission: int = 10, time_filter: str = 'all') -> List[RedditPostCaption]:
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

        # TODO: if submission.id already in JSON, skip it.
        submission_generator = self.reddit_client.subreddit(subreddit_name).top(limit=num_submissions, time_filter=time_filter)
        submissions = [sub for sub in submission_generator]
        print(f'Received {len(submissions)} submissions from subreddit {subreddit_name}')

        submissions = list(filter(lambda sub: self.crawler_data_params.filter_post(sub), submissions))
        print(f'{len(submissions)} submissions left after filtering')

        for submission in submissions:
            print('--------------------------------------------------')
            print(f'Title: {submission.title} (https://reddit.com{submission.permalink})')
            print(f'Self Text: {submission.selftext}')
            print(f'Image: {submission.url}')
            print('--------------------------------------------------')
            comments = submission.comments
            print(f'\nComments ({submission.num_comments}): \n\n')

            images = SubRedditCrawler._ingest_images(submission=submission, image_dir=image_dir) if self.crawler_data_params.ingest_images else None
            print(images)

            if any([sub.get('submission_id') == submission.id for sub in existing_data]):
                print(f"{submission.id} already exists in dataset!")
                continue

            # Grab the specified top level comments
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
                f"Going through {len(top_level_comments)} top level comments and {len(images) if images else 0} images for submission {submission.id}")

            # Output the top level comments
            for top_level_comment in sorted(top_level_comments, key=lambda c: c.score, reverse=True):
                if images:
                    for image in images:
                        image_captions.append(
                            RedditPostCaption(
                                submission_id=submission.id,
                                comment_id=top_level_comment.id,
                                title=submission.title,
                                self_text=submission.selftext,
                                comment=top_level_comment.body,
                                image_width=image.get("width"),
                                image_height=image.get("height"),
                                image_download_path=image.get("image_path"),
                                image_url=submission.url
                            )
                        )
                else:
                    image_captions.append(
                        RedditPostCaption(
                            submission_id=submission.id,
                            comment_id=top_level_comment.id,
                            self_text=submission.selftext,
                            title=submission.title,
                            comment=top_level_comment.body,
                        )
                    )

        print(f'Ingested {len(image_captions)} NEW RedditPostCaptions from subreddit {subreddit_name}')
        return image_captions

    @staticmethod
    def _ingest_images(submission, image_dir) -> Optional[List[Dict[str, str]]]:
        image_exists = lambda image_path: Path(image_path).exists()
        image_paths = [f'{image_dir}/{submission.id}_{key}.jpg' for key in submission.media_metadata.keys()] if hasattr(
            submission, 'media_metadata') else [f'{image_dir}/{submission.id}.jpg']
        print(f'Expected Images: {image_paths}')

        images = []

        if all([image_exists(image_path) for image_path in image_paths]):
            print(f"All {len(image_paths)} images for {submission.id} already downloaded. Simply returning the local reference...")

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

        return images

    def ingest_subreddit_as_conversation_json(self, subreddit_name: str, output_path: str, num_submissions: Optional[int] = None, comments_per_submission: int = 10):
        dataset_types = ["augmented"]

        # Ensure the output directories exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for dataset_type in dataset_types:
            full_path = output_path + '/' + dataset_type
            if not os.path.exists(full_path):
                os.makedirs(full_path)

        # Ensure the output files exists
        augmented_output_path = os.path.join(output_path, 'augmented', 'full_dataset.json')
        # basic_output_path = os.path.join(output_path, 'full_dataset.json')
        os.makedirs(os.path.dirname(augmented_output_path), exist_ok=True)
        # os.makedirs(os.path.dirname(basic_output_path), exist_ok=True)

        # Set up the image directory
        image_dir = output_path + '/images'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        reddit_post_captions = []
        for time_filter in ["all", "day", "hour", "month", "week", "year"]:
            try:
                new_posts = self._ingest_subreddit(subreddit_name, output_path, num_submissions, comments_per_submission, time_filter)
                print(f'Have {len(reddit_post_captions)} old+new image-caption pairs in total from subreddit {subreddit_name} for {time_filter} time')
                reddit_post_captions += new_posts
            except TooManyRequests:
                print('Too many requests. Received a 429. Sleeping for 60 seconds and then continuing')
                time.sleep(60)

        entries = [self.crawler_data_params.transform_post(post) for post in set(reddit_post_captions)]
        print(f'{len(entries)} total entries after de-duplicating')

        for prompt_type in dataset_types:
            full_dataset_path = os.path.join(output_path, prompt_type, 'full_dataset.json')

            # Load the existing data
            existing_data = []
            if os.path.exists(full_dataset_path):
                with open(full_dataset_path, 'r') as json_file:
                    existing_data = json.load(json_file)
                print(f'Loaded {len(existing_data)} existing image-caption pairs from {full_dataset_path}')

            # Add the new data to the existing data
            print(f'Adding {len(entries)} new image-caption pairs to the existing {len(existing_data)} pairs')
            full_ds = existing_data + [p.to_json() for p in entries]

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
