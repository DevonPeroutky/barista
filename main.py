from scripts.download_data import SubRedditCrawler

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Ingest image-caption pairs from subreddit
    # SubRedditCrawler().ingest_subreddit_as_csv('RoastMe', num_submissions=None)
    SubRedditCrawler().ingest_subreddit_as_conversation_json('RoastMe', output_path="datasets/conversational/", num_submissions=5)
