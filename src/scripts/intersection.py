import argparse
import json

# Load the JSON files
def determine_intersection(file1: str, file2: str):
    with open(file1, 'r') as f:
        data1 = json.load(f)

    with open(file2, 'r') as f:
        data2 = json.load(f)

    # Extract the submission_id from each object
    submission_ids1 = {item['submission_id'] for item in data1}
    submission_ids2 = {item['submission_id'] for item in data2}

    # Find the intersection
    common_submission_ids = submission_ids1.intersection(submission_ids2)

    # Print the common submission_ids
    for submission_id in common_submission_ids:
        print(submission_id)


if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description='Determine the intersection of two JSON files')
    # Add argument for file1 and file2
    parser.add_argument('--file1', type=str, help='The first JSON file', required=False, default="/home/devonperoutky/dataset/augmented_roast/checkpoints/reddit_post_captions.json")
    parser.add_argument('--file2', type=str, help='The second JSON file', required=False, default="/home/devonperoutky/dataset/augmented_roast/augmented/full_dataset.json")


    args = parser.parse_args()
    determine_intersection(args.file1, args.file2)


