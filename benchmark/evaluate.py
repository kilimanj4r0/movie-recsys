import argparse
import pickle
from rectools.metrics.scoring import calc_metrics
from rectools.metrics import F1Beta, MeanInvUserFreq, MAP, MRR, Serendipity
from rectools import Columns
from tabulate import tabulate


K = 10

METRICS_NAME = {
    'F1Beta': F1Beta,
    'MRR': MRR,
    'MAP': MAP,
    'Novelty': MeanInvUserFreq,
    'Serendipity': Serendipity
}
METRICS = {}
for metric_name, metric in METRICS_NAME.items():
    for k in (1, 5, 10):
        METRICS[f'{metric_name}@{k}'] = metric(k=k)

TEST_DATA_DIR = './data/'
DATA_FILENAMES = [f'u{t}.{split}' for t in ['a', 'b'] for split in ['base', 'test']]
DATA = {}

def print_metrics_table(metrics_dict):
    table = []

    for metric_name, metric_value in metrics_dict.items():
        table.append([metric_name, metric_value])

    print(tabulate(table))


def test_model(model, folds=['a', 'b']):
    total_metrics = {k: 0 for k in METRICS.keys()}
    for fold in folds:
        fold_data = load_data(fold)
        fold_metrics = train_and_evaluate(model, fold_data)
        for metric_name, metric_value in fold_metrics.items():
            total_metrics[metric_name] += metric_value
        
        print(f"Fold {fold}:")
        print_metrics_table(fold_metrics)
        print()
    
    average_metrics = {metric_name: total_value / len(folds) for metric_name, total_value in total_metrics.items()}
    print(f"Average across test folds:")
    print_metrics_table(average_metrics)


def train_and_evaluate(model, fold_data):
    train, test, train_df, test_df = fold_data
    model.fit(train)
    recos = model.recommend(
        users=train_df[Columns.User].unique(),
        dataset=train,
        k=K,
        filter_viewed=True,
    )
    return calc_metrics(
        METRICS,
        reco=recos,
        interactions=test_df,
        prev_interactions=train_df,
        catalog=train_df[Columns.Item].unique()
    )


def load_data(fold):
    return DATA[f'u{fold}']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a recommendation model.")
    parser.add_argument('--model_path',
                        default='../models/best_model.pickle',
                        help='Path to the model pickle file'
                        )
    args = parser.parse_args()

    for i in range(0, len(DATA_FILENAMES), 2):
        base_filename, test_filename = DATA_FILENAMES[i:i+2]
        data_title = base_filename.split('.')[0]
        with open(TEST_DATA_DIR + base_filename + '.pickle', 'rb') as base:
            with open(TEST_DATA_DIR + test_filename + '.pickle', 'rb') as test:
                with open(TEST_DATA_DIR + base_filename + '.df.pickle', 'rb') as base_df:
                    with open(TEST_DATA_DIR + test_filename + '.df.pickle', 'rb') as test_df:
                        DATA[data_title] = (pickle.load(base), pickle.load(test), pickle.load(base_df), pickle.load(test_df))

    model_path = args.model_path

    with open(model_path, 'rb') as pickle_file:
        best_model = pickle.load(pickle_file)

    test_model(best_model)
