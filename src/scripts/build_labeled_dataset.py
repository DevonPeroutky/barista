import pandas as pd


class BuildLabeledDataset:
    def __init__(self, dataset, labels, output):
        self.dataset = dataset
        self.labels = labels
        self.output = output

    def build(self):
        # Read dataset
        df = pd.read_csv(self.dataset)

        # Read labels
        labels = pd.read_csv(self.labels)

        # Merge dataset and labels
        df = pd.merge(df, labels, on='id')

        # Save dataset
        df.to_csv(self.output, index=False)

        return df

    def build_labels(self):
        # Read dataset
        df = pd.read_csv(self.dataset)

        return df