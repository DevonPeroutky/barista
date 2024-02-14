import json
import os

class QuestionAnswerDatasetCreator:

    @staticmethod
    def create_vqa_from_dataset(dataset_path, output_path):
        # Read dataset from json file as JSON array
        with open(dataset_path, 'r') as file:
            dataset = json.load(file)

        assert dataset is not None

        # iterate over the dataset with indexes and create a list of dictionaries
        vqa = []

        questions = [{ "question_id": submission['id'], "image": submission['image'], "category": "detail", "text": submission['conversations'][0]['value']  } for submission in dataset]
        answers = [{ "question_id": submission['id'], "category": "detail", "text": submission['conversations'][1]['value'] } for submission in dataset]

        print(len(questions))
        print(len(answers))

        print(questions[0:3])
        print(answers[0:3])

        # make sure output path exists if it doesn't create it
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # write the list of dictionaries to a json file
        with open(output_path + 'questions.json', 'w') as file:
            json.dump(questions, file, indent=4)

        # write the list of dictionaries to a json file
        with open(output_path + 'answers.json', 'w') as file:
            json.dump(answers, file, indent=4)

        return vqa


if __name__ == '__main__':

    QuestionAnswerDatasetCreator.create_vqa_from_dataset('datasets/augmented/full_dataset.json', 'eval_vqa/')
    print("Done")
