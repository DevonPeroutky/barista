from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

from praw.models import Submission

from src.clients import OpenAIVisionAssistant, ClaudeVisionAssistant, VisionAssistant


@dataclass
class RedditPostCaption:
    submission_id: str
    comment_id: str
    title: str
    comment: str
    permalink: Optional[str] = None
    self_text: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_download_path: Optional[str] = None
    image_url: Optional[str] = None

    def _id(self):
        return f"{self.submission_id}_{self.comment_id}_{self.image_url}"

    def __eq__(self, other):
        return isinstance(other, RedditPostCaption) and self.submission_id == other.submission_id and self.comment_id == other.comment_id and self.image_url == other.image_url

    def __hash__(self):
        return hash((self.submission_id, self.comment_id, self.image_url))

    @staticmethod
    def from_json(json_data):
        return RedditPostCaption(
            submission_id=json_data.get("submission_id"),
            comment_id=json_data.get("comment_id"),
            title=json_data.get("title"),
            comment=json_data.get("comment"),
            self_text=json_data.get("self_text"),
            permalink=json_data.get("permalink"),
            image_width=json_data.get("image_width"),
            image_height=json_data.get("image_height"),
            image_download_path=json_data.get("image_download_path"),
            image_url=json_data.get("image_url"),
        )

    def to_json(self, prompt_type="basic"):
        return {
            "id": f"{self.submission_id}_{self.comment_id}",
            "submission_id": self.submission_id,
            "permalink": self.permalink,
            "title": self.title,
            "image": self.image_download_path,
            "comment_id": self.comment_id,
            "comment": self.comment,
            "image_url": self.image_url,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_download_path": self.image_download_path,
            "self_text": self.self_text,
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


@dataclass
class QuestionAnswerEntry:
    id: str
    title: str
    submission_id: str
    comment_id: str
    human_prompt: str
    ai_response: str
    self_text: Optional[str] = None
    image_url: Optional[str] = None
    image_download_path: Optional[str] = None

    def __hash__(self):
        return hash((self.submission_id, self.comment_id, self.image_url))

    def __eq__(self, other):
        return isinstance(other, QuestionAnswerEntry) and self.submission_id == other.submission_id and self.comment_id == other.comment_id and self.image_url == other.image_url

    def _id(self):
        return f"{self.submission_id}_{self.comment_id}_{self.image_url}"

    # {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
    # {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris", "weight": 0}, {"role": "user", "content": "Can you be more sarcastic?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already.", "weight": 1}]}
    def to_openai_format(self):
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a clever and mean therapist, as well as a hilarious comedian, who gives advice that is meant to be humorous. You don't care about the well-being of the person asking the questions. You only care about giving the funniest advice to the questions asked. Be clever and witty and don't ramble. Don't introduce your answer, just give a list of witty and humorous responses."
                },
                {
                    "role": "user",
                    "content": self.human_prompt
                },
                {
                    "role": "assistant",
                    "content": self.ai_response
                }
            ]
        }

    def to_json(self):
        return {
            "id": self.id,
            "submission_id": self.submission_id,
            "image": self.image_download_path,
            "title": self.title,
            "self_text": self.self_text,
            "comment_id": self.comment_id,
            "conversations": [
                {
                    "from": "human",
                    "value": self.human_prompt
                },
                {
                    "from": "gpt",
                    "value": self.ai_response
                }
            ]
        }

    @staticmethod
    def from_json(json_data):
        # Extract fields from the JSON data
        id = json_data["id"]
        submission_id = json_data["submission_id"]
        image_download_path = json_data.get("image")  # Using get to handle optional fields
        title = json_data["title"]
        self_text = json_data.get("self_text")
        comment_id = json_data["comment_id"]

        # Assuming conversations list contains human prompt and AI response in order
        conversations = json_data.get("conversations", [])
        human_prompt = conversations[0]["value"] if conversations else ""
        ai_response = conversations[1]["value"] if len(conversations) > 1 else ""

        return QuestionAnswerEntry(
            id=id,
            title=title,
            submission_id=submission_id,
            comment_id=comment_id,
            human_prompt=human_prompt,
            ai_response=ai_response,
            self_text=self_text,
            image_download_path=image_download_path
        )


class CrawlerDataParams(ABC):
    ingest_images = True
    images_required = False

    @staticmethod
    def filter_submission(submission: Submission) -> bool:
        return True

    @staticmethod
    def filter_post(post: RedditPostCaption) -> bool:
        return True

    @staticmethod
    @abstractmethod
    def transform_post(post: RedditPostCaption) -> QuestionAnswerEntry:
        pass


class RoastCrawlerParams(CrawlerDataParams):
    def __init__(self, vision_assistant: VisionAssistant):
        self.vision_assistant = vision_assistant
        self.ingest_images = True
        self.images_required = True
        # add a key-value dictionary with string -> string mapping
        self.post_image_captions = {}

    @staticmethod
    def filter_post(post: RedditPostCaption) -> bool:
        return post.image_url is not None

    def transform_post(self, post: RedditPostCaption) -> QuestionAnswerEntry:
        submission_image_key = f'{post.submission_id}_{post.image_url}'
        print("-"*10)
        print(submission_image_key)
        print(post.permalink)
        caption = self.post_image_captions.get(submission_image_key)

        if not caption:
            caption = self.vision_assistant.describe_person_in_image(image_url=post.image_url)
            self.post_image_captions[submission_image_key] = caption

        reddit_title_with_punc = f'{post.title}{"" if post.title.endswith(".") or post.title.endswith("!") or post.title.endswith("?") else "."}'
        full_prompt = f'{caption} They have this to say "{reddit_title_with_punc}"'

        return QuestionAnswerEntry(
            id=f"{post.submission_id}_{post.comment_id}",
            title=post.title,
            human_prompt=f'<image>\n {full_prompt} Roast this person.',
            ai_response=post.comment,
            submission_id=post.submission_id,
            comment_id=post.comment_id,
            image_url=post.image_url,
            image_download_path=post.image_download_path
        )


class TextOnlyCrawlerParams(CrawlerDataParams):
    def __init__(self):
        self.ingest_images = False

    @staticmethod
    def filter_submission(submission: Submission) -> bool:
        # Filter for question-answer pairs in r/UnethicalLifeProTips
        if "ulpt request" in submission.title.lower():
            print("YES")
            return True
        else:
            print("NOTHING")
            return False

    @staticmethod
    def strip_prefix(title: str):
        title = title.strip()
        title = title.lstrip("ULPT")
        title = title.lstrip("ulpt")
        title = title.lstrip("Ulpt")
        title = title.strip()
        title = title.lstrip("Request")
        title = title.lstrip("REQUEST")
        title = title.lstrip("request")
        title = title.strip()
        title = title.lstrip(":")
        title = title.strip()
        title = title.lstrip("-")
        title = title.strip()
        return title

    @staticmethod
    def transform_post(post: RedditPostCaption) -> QuestionAnswerEntry:
        cleaned_title = TextOnlyCrawlerParams.strip_prefix(title=post.title)
        return QuestionAnswerEntry(
            id=f"{post.submission_id}_{post.comment_id}",
            title=post.title,
            self_text=post.self_text,
            human_prompt=f'{cleaned_title} {post.self_text}' if post.self_text else cleaned_title,
            ai_response=post.comment,
            submission_id=post.submission_id,
            comment_id=post.comment_id,
            image_download_path=post.image_download_path
        )

