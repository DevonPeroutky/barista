from scripts.download_data import SubRedditCrawler
from models import RoastCrawlerParams, TextOnlyCrawlerParams

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Ingest image-caption pairs from subreddit
    # SubRedditCrawler().ingest_subreddit_as_csv('RoastMe', num_submissions=None)
    # SubRedditCrawler(params=RoastCrawlerParams()).ingest_subreddit_as_conversation_json('roastme', output_path="/home/devonperoutky/temp", num_submissions=3)
    SubRedditCrawler(params=TextOnlyCrawlerParams()).ingest_subreddit_as_conversation_json('shittyadvice', output_path="/home/devonperoutky/temp/advice", num_submissions=10)
