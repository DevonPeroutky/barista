from src.scripts.download_data import SubRedditCrawler
from src.clients import OpenAIVisionAssistant, ClaudeVisionAssistant
from src.models import RoastCrawlerParams

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    openAIVisionassistant = OpenAIVisionAssistant()
    claudeAIVisionassistant = ClaudeVisionAssistant(model="claude-3-opus-20240229")

    # Ingest image-caption pairs from subreddit
    # SubRedditCrawler().ingest_subreddit_as_csv('RoastMe', num_submissions=None)
    # SubRedditCrawler(params=RoastCrawlerParams(vision_assistant=claudeAIVisionassistant)).ingest_subreddit_as_conversation_json('roastme', output_path="/home/devonperoutky/dataset/augmented_roast", num_submissions=3000)
    # SubRedditCrawler(params=TextOnlyCrawlerParams()).ingest_subreddit_as_conversation_json('roastme', output_path="/home/devonperoutky/dataset/augmented_roast", num_submissions=5000)
    SubRedditCrawler(params=RoastCrawlerParams(vision_assistant=openAIVisionassistant)).continue_from_checkpoint(output_path="/home/devonperoutky/dataset/augmented_roast")


def prepare_openai_dataset(dataset_path):
    # Load the dataset from the json file at dataset_path
    # Then seriallize the json to QuestionAnswerEntries
    # Then transform the conversations of the QuestionAnswerEntries to the format that OpenAI's GPT-3 API expects
    # Here is the example format:
    # {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
    # Then save the transformed dataset to a new json file
    import json
    from src.models import QuestionAnswerEntry
    from src.clients import OpenAIVisionAssistant

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    entries = [QuestionAnswerEntry.from_json(entry) for entry in dataset]
    openAIVisionassistant = OpenAIVisionAssistant()
    for entry in entries:
        entry.transform_conversation_to_openai_format(vision_assistant=openAIVisionassistant)

    with open(dataset_path.replace('.json', '_openai.json'), 'w') as f:
        json.dump([entry.to_json() for entry in entries], f)
