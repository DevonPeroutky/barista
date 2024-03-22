from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

from praw.models import Submission


@dataclass
class RedditPostCaption:
    submission_id: str
    comment_id: str
    title: str
    comment: str
    self_text: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_download_path: Optional[str] = None
    image_url: Optional[str] = None

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


@dataclass
class QuestionAnswerEntry:
    id: str
    title: str
    submission_id: str
    comment_id: str
    human_prompt: str
    ai_response: str
    self_text: Optional[str] = None
    image_download_path: Optional[str] = None

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


class CrawlerDataParams(ABC):
    ingest_images = True
    images_required = False

    @staticmethod
    def filter_post(submission: Submission) -> bool:
        return True

    @staticmethod
    @abstractmethod
    def transform_post(post: RedditPostCaption) -> QuestionAnswerEntry:
        pass


class RoastCrawlerParams(CrawlerDataParams):
    def __init__(self):
        self.ingest_images = True
        self.images_required = True

    @staticmethod
    def transform_post(post: RedditPostCaption) -> QuestionAnswerEntry:
        return QuestionAnswerEntry(
            id=f"{post.submission_id}_{post.comment_id}",
            title=post.title,
            human_prompt=f'<image>\n{post.title}{"" if post.title.endswith(".") or post.title.endswith("!") or post.title.endswith("?") else "."} Write an insult about this person.',
            ai_response=post.comment,
            submission_id=post.submission_id,
            comment_id=post.comment_id,
            image_download_path=post.image_download_path
        )


class TextOnlyCrawlerParams(CrawlerDataParams):
    def __init__(self):
        self.ingest_images = False

    @staticmethod
    def filter_post(submission: Submission) -> bool:
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

