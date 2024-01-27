import PIL
import praw
import itertools
from PIL import Image
import re

import requests
import shutil
import os


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

    def ingest_subreddit(self, subreddit_name, num_submissions=3, comments_per_submission=10):

        # Set up the image directory
        image_dir = './images'
        ds_dir = './datasets'
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # Write the header to the file only if file doesn't exist already
        if not os.path.exists(f'{ds_dir}/full_ds.csv'):
            with open(f'{ds_dir}/full_ds.csv', 'w+') as out_file:
                out_file.write(f'submission_id,comment_id,comment,width,height,image_url,image_path\n')

        for submission in self.reddit_client.subreddit(subreddit_name).top(limit=num_submissions):

            print('--------------------------------------------------')
            print(f'Title: {submission.title} (https://reddit.com{submission.permalink})')
            print(f'Image: {submission.url}')
            print('--------------------------------------------------')
            comments = submission.comments
            print(f'\nComments ({submission.num_comments}): \n\n')

            # if submission.id in the csv file, skip it:
            if submission.id in open(f'{ds_dir}/full_ds.csv').read():
                print("Submission already in dataset, skipping")
                continue

            # Check if submission has media_metadata attribute
            images = []

            if hasattr(submission, 'media_metadata'):
                print("Submission has media metadata, downloading the full gallery of images")

                # For each key-value pair in the media_metadata dictionary
                for key, value in submission.media_metadata.items():
                    # Get the url of the image
                    image_url = value.get('s').get('u')
                    print("Downloading image from " + image_url)
                    response = requests.get(image_url, stream=True)
                    with open(f'{image_dir}/{submission.id}_{key}.jpg', 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)

                    print(f'Downloaded image to {image_dir}/{key}.jpg')

                    # Get height and width of image in pixels
                    images.append({
                        'width': value.get('s').get('x'),
                        'height': value.get('s').get('y'),
                        'image_path': f'{image_dir}/{submission.id}_{key}.jpg',
                    })
                    del response
            else:
                print("Downloading image from " + submission.url)
                response = requests.get(submission.url, stream=True)
                with open(f'{image_dir}/{submission.id}.jpg', 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)

                print(f'Downloaded image to {image_dir}/{submission.id}.jpg')

                try:
                    # Get height and width of image in pixels
                    im = Image.open(f'{image_dir}/{submission.id}.jpg')
                    width, height = im.size
                    images.append({
                        'width': width,
                        'height': height,
                        'image_path': f'{image_dir}/{submission.id}.jpg',
                    })
                    del im
                    del response
                except PIL.UnidentifiedImageError as e:
                    print("Error opening image, skipping submission")
                    continue

            # Grap the specified top level comments
            top_level_comments = filter(lambda c: not isinstance(c, praw.models.MoreComments), comments[:comments_per_submission])

            # Filter out top-level comments that contain a markdown url in the comment body using regex
            top_level_comments = list(filter(lambda c: not re.search(r'\[.*\]\(.*\)', c.body), top_level_comments))

            # Filter out top-level comments that contain a markdown url in the comment body using regex
            top_level_comments = list(filter(lambda c: c.body is not '[deleted]', top_level_comments))

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

            print(f"Going through {len(top_level_comments)} top level comments and {len(images)} images for submission {submission.id}")
            out_file = open(f'{ds_dir}/full_ds.csv', 'a')

            # Print the top level comments
            for top_level_comment in sorted(top_level_comments, key=lambda c: c.score, reverse=True):
                for image in images:
                    line = f'{submission.id},{top_level_comment.id},{top_level_comment.body},{image.get("width")},{image.get("height")},{submission.url},{image.get("image_path")}'
                    print(line)
                    out_file.write(line+'\n')

            out_file.close()