from dataclasses import dataclass
from typing import List

import PIL
import praw
import itertools
from PIL import Image
import re

import requests
import shutil
import os
import json


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

    def to_json(self, prompt_type="basic"):
        return {
            "id": self.submission_id,
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

    def ingest_subreddit_as_csv(self, subreddit_name, num_submissions=None, comments_per_submission=10):
        out_path = './datasets/raw/'
        image_captions = self._ingest_subreddit(subreddit_name, num_submissions, comments_per_submission)
        print(f'Ingested {len(image_captions)} image-caption pairs from subreddit {subreddit_name}')

        out_file = open(f'{out_path}/full_ds.csv', 'a')
        for image_caption in image_captions:
            print(image_caption.to_csv())
            out_file.write(image_caption.to_csv() + '\n')

        out_file.close()

    def _ingest_subreddit(self, subreddit_name, num_submissions=None, comments_per_submission=10) -> List[RedditPostCaption]:

        # Set up the image directory
        image_dir = './images'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        image_captions = []

        for submission in self.reddit_client.subreddit(subreddit_name).top(limit=num_submissions):
            print('--------------------------------------------------')
            print(f'Title: {submission.title} (https://reddit.com{submission.permalink})')
            print(f'Self Text: {submission.selftext}')
            print(f'Image: {submission.url}')
            print('--------------------------------------------------')
            comments = submission.comments
            print(f'\nComments ({submission.num_comments}): \n\n')

            # TODO: if submission.id alread in JSON, skip it.

            # Check if submission has media_metadata attribute
            images = []

            if hasattr(submission, 'media_metadata'):
                print("Submission has media metadata, downloading the full gallery of images")

                # For each key-value pair in the media_metadata dictionary
                for key, value in submission.media_metadata.items():
                    # Get the url of the image
                    image_url = value.get('s').get('u')
                    image, width, height, download_path = SubRedditCrawler.download_image(image_url,
                                                                                          f'{image_dir}/{submission.id}_key.jpg')
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

        return image_captions

    def ingest_subreddit_as_conversation_json(self, subreddit_name, output_path, num_submissions=None, comments_per_submission=10):
        reddit_post_captions = self._ingest_subreddit(subreddit_name, num_submissions, comments_per_submission)
        print(f'Ingested {len(reddit_post_captions)} image-caption pairs from subreddit {subreddit_name}')

        # Split image_captions into train, test, and validation sets
        train_percentage, test_percentage, validation_percentage = .8, .1, .1
        assert train_percentage + test_percentage + validation_percentage == 1

        train_size = int(train_percentage * len(reddit_post_captions))
        test_size = int(test_percentage * len(reddit_post_captions))

        train_split = reddit_post_captions[0:train_size]
        test_split = reddit_post_captions[train_size:train_size + test_size]
        validation_split = reddit_post_captions[train_size+test_size:]

        # Ensure the output directories exists
        augmented_output_path = os.path.join(output_path, 'augmented', 'full_dataset.json')
        basic_output_path = os.path.join(output_path, 'basic', 'full_dataset.json')
        os.makedirs(os.path.dirname(augmented_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(basic_output_path), exist_ok=True)

        for prompt_type in ["basic", "augmented"]:
            json_output_path = os.path.join(output_path, prompt_type, 'full_dataset.json')
            with open(json_output_path, 'w') as json_file:
                json.dump([p.to_json(prompt_type) for p in reddit_post_captions], json_file, indent=4)
                print(f'Saved {len(reddit_post_captions)} image-caption pairs to {json_output_path}')

            json_output_path = os.path.join(output_path, prompt_type, 'train_dataset.json')
            with open(json_output_path, 'w') as json_file:
                json.dump([p.to_json(prompt_type) for p in train_split], json_file, indent=4)
                print(f'Saved {len(train_split)} image-caption pairs to {json_output_path}')

            json_output_path = os.path.join(output_path, prompt_type, 'test_dataset.json')
            with open(json_output_path, 'w') as json_file:
                json.dump([p.to_json(prompt_type) for p in test_split], json_file, indent=4)
                print(f'Saved {len(test_split)} image-caption pairs to {json_output_path}')

            json_output_path = os.path.join(output_path, prompt_type, 'validation_dataset.json')
            with open(json_output_path, 'w') as json_file:
                json.dump([p.to_json(prompt_type) for p in validation_split], json_file, indent=4)
                print(f'Saved {len(validation_split)} image-caption pairs to {json_output_path}')


    @staticmethod
    def download_image( image_url, download_path):
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
            print("Error opening image, skipping submission")
            return None, None, None, None