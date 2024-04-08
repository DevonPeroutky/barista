import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import PIL
import anthropic
import openai
import praw
from PIL import Image
import re

import requests
import shutil
import os
import json

from prawcore import TooManyRequests

from src.models import RedditPostCaption, CrawlerDataParams, QuestionAnswerEntry


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

    # TODO:
    # 1. Make image downloading optional
    # 2. Optional transformation of the title + selftext --> question?
    def _ingest_subreddit(self, subreddit_name, output_path, num_submissions: int = None,
                          comments_per_submission: int = 10, time_filter: str = 'all',
                          existing_data: List[QuestionAnswerEntry] = []) -> List[RedditPostCaption]:
        assert time_filter in ["all", "day", "hour", "month", "week", "year"], f'Invalid time filter: {time_filter}'

        print(f'\n\n-------- Ingesting top {num_submissions} submissions from {subreddit_name} for {time_filter}')

        image_dir = output_path + '/images'
        image_captions = []

        # Get the list of filenames in the images directory
        existing_images = os.listdir(image_dir)
        print(f'Loaded {len(existing_images)} existing downloaded images from {image_dir}')

        # TODO: if submission.id already in JSON, skip it.
        submission_generator = self.reddit_client.subreddit(subreddit_name).top(limit=num_submissions,
                                                                                time_filter=time_filter)
        submissions = [sub for sub in submission_generator]
        print(f'Received {len(submissions)} submissions from subreddit {subreddit_name}')

        submissions = list(filter(lambda sub: self.crawler_data_params.filter_submission(sub), submissions))
        print(f'{len(submissions)} submissions left after filtering')

        for submission in submissions:
            print('--------------------------------------------------')
            print(f'Title: {submission.title} (https://reddit.com{submission.permalink})')
            print(f'Self Text: {submission.selftext}')
            print(f'Image: {submission.url}')
            print('--------------------------------------------------')
            comments = submission.comments
            print(f'\nComments ({submission.num_comments}): \n\n')

            images = SubRedditCrawler._ingest_images(submission=submission,
                                                     image_dir=image_dir) if self.crawler_data_params.ingest_images else None
            print(images)

            if any([sub.submission_id == submission.id for sub in existing_data]):
                print(f"{submission.id} already exists in dataset!")
                continue

            # Grab the specified top level comments
            top_level_comments = filter(lambda c: not isinstance(c, praw.models.MoreComments),
                                        comments[:comments_per_submission])

            # Filter out top-level comments that contain a markdown url in the comment body using regex
            top_level_comments = list(filter(lambda c: not re.search(r'\[.*\]\(.*\)', c.body), top_level_comments))

            # Filter out top-level comments that contain a markdown url in the comment body using regex
            top_level_comments = list(
                filter(lambda c: '[deleted]' not in c.body and '[removed]' not in c.body, top_level_comments))

            # Filter out the comments that start with "OP's Bio:" or "OP's bio:"
            top_level_comments = list(filter(lambda c: not c.body.startswith("OP's Bio:"), top_level_comments))

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
                                image_download_path=image.get("local_image_path"),
                                image_url=image.get("image_url")
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
        images = []

        # Submission has a gallery of images
        if hasattr(submission, 'media_metadata'):
            image_paths = [f'{image_dir}/{submission.id}_{key}.jpg' for key in submission.media_metadata.keys()]
            image_urls = [value.get('s').get('u') for value in submission.media_metadata.values()]

            for (url, path) in zip(image_urls, image_paths):
                if image_exists(path):
                    print(f"Image {path} already downloaded. Simply returning the local reference...")
                    image = Image.open(path)
                    images.append({
                        'width': image.width,
                        'height': image.height,
                        'local_image_path': path,
                        'image_url': url,
                    })
                else:
                    image, width, height, download_path = SubRedditCrawler._download_image(url, path)
                    if image:
                        images.append({
                            'width': width,
                            'height': height,
                            'local_image_path': path,
                            'image_url': url,
                        })
        # Submission has a single image that has already been downloaded
        elif image_exists(f'{image_dir}/{submission.id}.jpg'):
            print(f"Image {submission.id}.jpg already downloaded. Simply returning the local reference...")
            image = Image.open(f'{image_dir}/{submission.id}.jpg')
            images.append({
                'width': image.width,
                'height': image.height,
                'local_image_path': f'{image_dir}/{submission.id}.jpg',
                'image_url': submission.url,
            })
        # Submission has a single image that has not been downloaded
        else:
            image, width, height, download_path = SubRedditCrawler._download_image(submission.url,
                                                                                   f'{image_dir}/{submission.id}.jpg')
            if image:
                images.append({
                    'width': width,
                    'height': height,
                    'local_image_path': download_path,
                    'image_url': submission.url,
                })

        return images

    def ingest_subreddit_as_conversation_json(self, subreddit_name: str, output_path: str,
                                              num_submissions: Optional[int] = None, comments_per_submission: int = 10):
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

        # Load the existing data
        existing_data = []
        file_path = f'{output_path}/augmented/full_dataset.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                existing_data = json.load(json_file)
            print(f'Loaded {len(existing_data)} existing image-caption pairs from {file_path}')

        # Load existing data as QuestionAnswer objects
        existing_data: List[QuestionAnswerEntry] = [QuestionAnswerEntry.from_json(json_data=post) for post in
                                                    existing_data]

        reddit_post_captions = []
        for time_filter in ["all", "day", "month", "week", "year"]:
            try:
                new_posts = self._ingest_subreddit(subreddit_name, output_path, num_submissions,
                                                   comments_per_submission, time_filter, existing_data)
                print(
                    f'Have {len(new_posts)} new image-caption pairs from subreddit {subreddit_name} for {time_filter} time')
                reddit_post_captions += new_posts
            except TooManyRequests:
                print('Too many requests. Received a 429. Sleeping for 60 seconds and then continuing')
                time.sleep(60)

        # Filter reddit_post_captions
        print(f"{len(reddit_post_captions)} new image-caption pairs ingested")
        reddit_post_captions = list(filter(self.crawler_data_params.filter_post, reddit_post_captions))
        print(f"{len(reddit_post_captions)} new image-caption pairs after filtering")

        # Save the new data to the checkpoint before augementing and transforming them into QuestionAnswer objects
        if not os.path.exists(f'{output_path}/checkpoints'):
            os.makedirs(f'{output_path}/checkpoints')

        with open(f'{output_path}/checkpoints/reddit_post_captions.json', 'w') as json_file:
            json.dump([p.to_json() for p in reddit_post_captions], json_file, indent=4)

        # Transform the RedditPostCaptions into QuestionAnswer objects
        return
        # self.continue_from_checkpoint(output_path=output_path)

    @staticmethod
    def _save_captions_to_dataset(dataset_path, dataset, prompt_type):
        print(f'Saving {len(dataset)} captions for prompt type {prompt_type} to {dataset_path}')

        # Save the new data to the dataset
        with open(dataset_path, 'w') as json_file:
            json.dump(dataset, json_file, indent=4)
            print(f'Saved {len(dataset)} image-caption pairs to {dataset_path}')

    @staticmethod
    def _download_image(image_url, download_path):
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

    @staticmethod
    def _update_checkpoint(reddit_post_captions: List[RedditPostCaption], entries: List[QuestionAnswerEntry], output_path: str):
        # Remove the newly processed from the checkpoint
        unprocessed_reddit_post_captions = list(filter(lambda p: p._id() not in set([e._id() for e in entries]), reddit_post_captions))
        print(f'{len(unprocessed_reddit_post_captions)} unprocessed')

        # Save the new data to the dataset
        with open(f'{output_path}/checkpoints/reddit_post_captions.json', 'w') as json_file:
            json.dump([p.to_json() for p in unprocessed_reddit_post_captions], json_file, indent=4)
        print(f'Saved {len(unprocessed_reddit_post_captions)} unprocessed image-caption pairs to checkpoint')

    def continue_from_checkpoint(self, output_path, num_to_process: Optional[int] = None):
        reddit_post_captions = []
        with open(f'{output_path}/checkpoints/reddit_post_captions.json', 'r') as json_file:
            reddit_post_captions = json.load(json_file)

        reddit_post_captions = [RedditPostCaption.from_json(json_data=post) for post in reddit_post_captions]

        # Filter reddit_post_captions
        print(f"{len(reddit_post_captions)} new image-caption pairs in the checkpoint")
        og_reddit_post_captions: List[RedditPostCaption] = list(set(list(filter(self.crawler_data_params.filter_post, reddit_post_captions))))
        reddit_post_captions = og_reddit_post_captions
        print(f"{len(reddit_post_captions)} new image-caption pairs after filtering and deduplicating")

        # Load the existing data
        existing_data = []
        file_path = f'{output_path}/augmented/full_dataset.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                existing_data = json.load(json_file)

        entries: List[QuestionAnswerEntry] = [QuestionAnswerEntry.from_json(json_data=post) for post in existing_data]
        print(f'Loaded {len(entries)} existing entries pairs from {file_path}')
        entries = list(set(entries))
        print(f'{len(entries)} entries after de-duplicating')

        if num_to_process:
            print(f"Only going to process the first {num_to_process} reddit_post_captions")
            reddit_post_captions = reddit_post_captions[0:num_to_process]

        errors = []
        for post in list(set(reddit_post_captions)):
            try:
                new_entry = self.crawler_data_params.transform_post(post)
                entries.append(new_entry)
                full_ds = [p.to_json() for p in list(set(entries))]
                SubRedditCrawler._save_captions_to_dataset(os.path.join(output_path, 'augmented', 'full_dataset.json'),
                                                           full_ds, 'augmented')
                self._update_checkpoint(reddit_post_captions=og_reddit_post_captions, entries=entries,
                                        output_path=output_path)
            except openai.RateLimitError as e:
                print("Rate limited. Going to just commit the current data and exit")
                full_ds = [p.to_json() for p in list(set(entries))]
                SubRedditCrawler._save_captions_to_dataset(os.path.join(output_path, 'augmented', 'full_dataset.json'),
                                                           full_ds, 'augmented')
                break
            except anthropic.RateLimitError as e:
                print("Rate limited. Going to just commit the current data and exit")
                full_ds = [p.to_json() for p in list(set(entries))]
                SubRedditCrawler._save_captions_to_dataset(os.path.join(output_path, 'augmented', 'full_dataset.json'),
                                                           full_ds, 'augmented')
                break
            except Exception as e:
                print(e)
                full_ds = [p.to_json() for p in list(set(entries))]
                errors.append(post)
                SubRedditCrawler._save_captions_to_dataset(os.path.join(output_path, 'augmented', 'full_dataset.json'),
                                                           full_ds, 'augmented')
                self._update_checkpoint(reddit_post_captions=og_reddit_post_captions, entries=entries,
                                        output_path=output_path)
                print("----> Ran into error. Going to just commit the current data and keep going")
                continue

        entries = list(set(entries))
        print(f'{len(entries)} total entries after adding and de-duplicating')
        full_ds = [p.to_json() for p in entries]
        SubRedditCrawler._save_captions_to_dataset(os.path.join(output_path, 'augmented', 'final_full_dataset.json'), full_ds, 'augmented')
        self._update_checkpoint(reddit_post_captions=og_reddit_post_captions, entries=entries, output_path=output_path)

        if errors:
            print(f"Could not process the {len(errors)} errors from the following posts:", errors)
