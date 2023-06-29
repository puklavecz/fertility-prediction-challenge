"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the predict_outcomes function. 

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

The script can be run from the command line using the following command:

python script.py input_path 

An example for the provided test is:

python script.py data/test_data_liss_2_subjects.csv
"""

import os
import sys
import argparse
import pandas as pd
from joblib import load

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.impute import KNNImputer

parser = argparse.ArgumentParser(description="Process and score data.")
subparsers = parser.add_subparsers(dest="command")

# Process subcommand
process_parser = subparsers.add_parser("predict", help="Process input data for prediction.")
process_parser.add_argument("input_path", help="Path to input data CSV file.")
process_parser.add_argument("--output", help="Path to prediction output CSV file.")

# Score subcommand
score_parser = subparsers.add_parser("score", help="Score (evaluate) predictions.")
score_parser.add_argument("prediction_path", help="Path to predicted outcome CSV file.")
score_parser.add_argument("ground_truth_path", help="Path to ground truth outcome CSV file.")
score_parser.add_argument("--output", help="Path to evaluation score output CSV file.")

args = parser.parse_args()


def predict_outcomes(df):
    """Process the input data and write the predictions."""

    # The predict_outcomes function accepts a Pandas DataFrame as an argument
    # and returns a new DataFrame with two columns: nomem_encr and
    # prediction. The nomem_encr column in the new DataFrame replicates the
    # corresponding column from the input DataFrame. The prediction
    # column contains predictions for each corresponding nomem_encr. Each
    # prediction is represented as a binary value: '0' indicates that the
    # individual did not have a child during 2020-2022, while '1' implies that
    # they did.
    
    # create variable
    count = 0
    def get_relationship_length(row, year=19):
        global count
        if year <= 8:
            return np.nan
        colname_year = 'cf' + '{:02d}'.format(year) + chr(year+89) + '028'
        # colname_partner = 'cf' + '{:02d}'.format(year) + chr(year+89) + '024'
        colname_partner = 'cf19l024'
        colname_same_partner = 'cf' + '{:02d}'.format(year) + chr(year+89) + '402'
        if pd.notna(row[colname_year]):
            return 2019 - row[colname_year]
        elif row[colname_partner] == 2:
            return 0
        elif colname_same_partner == 2:
            return 2019 - row[colname_year] if pd.notna(row[colname_year]) else row[colname_year]
        else:
            count += 1
            return get_relationship_length(row, year-1)

    df['RELLEN'] = [get_relationship_length(row[1]) for row in df.iterrows()]
    
    df["cf19l128"] = np.where(np.isnan(df["cf19l128"]), 2, df["cf19l128"])
    df["cf18k128"] = np.where(np.isnan(df["cf18k128"]), 2, df["cf18k128"])
    df["cf17j128"] = np.where(np.isnan(df["cf17j128"]), 2, df["cf17j128"])
    df["cf16i128"] = np.where(np.isnan(df["cf16i128"]), 2, df["cf16i128"])
    df["cf15h128"] = np.where(np.isnan(df["cf15h128"]), 2, df["cf15h128"])
    df["cf14g128"] = np.where(np.isnan(df["cf14g128"]), 2, df["cf14g128"])
    df["cf13f128"] = np.where(np.isnan(df["cf13f128"]), 2, df["cf13f128"])
    df["cf12e156"] = np.where(np.isnan(df["cf12e156"]), 2, df["cf12e156"])
    df["cf11d156"] = np.where(np.isnan(df["cf11d156"]), 2, df["cf11d156"])
    df["cf10c156"] = np.where(np.isnan(df["cf10c156"]), 2, df["cf10c156"])
    df["cf09b156"] = np.where(np.isnan(df["cf09b156"]), 2, df["cf09b156"])
    df["cf08a156"] = np.where(np.isnan(df["cf08a156"]), 2, df["cf08a156"])
    
    # Keep 
    keepcols = ['geslacht', 'leeftijd2019', 'herkomstgroep2019', 'cr19l089',
       'burgstat2019', 'woonvorm2019', 'cf19l024', 'cf19l025', 'cf19l030',
       'partner2019', 'RELLEN', 'cf19l183', 'aantalki2019', 'cf19l454',
       'oplmet2019', 'belbezig2019', 'nettohh_f2019', 'woning2019', 'cf19l128',
       'cf19l129', 'cf19l130', 'cp19k010', 'cp19k011', 'cf17j180', 'cf18k180',
       'cf19l180', 'cf19l483', 'cf19l484', 'cf19l485', 'cf19l486', 'cf19l487',
       'cf19l488', 'cf19l198', 'cf19l199', 'cf19l200', 'cf19l201', 'ch19l004',
       'ch19l018', 'ch19l021', 'ch19l022', 'ch19l229', 'ch19l219', 'cv19k012',
       'cv19k053', 'cv19k101', 'cv19k125', 'cv19k126', 'cv19k130', 'cv19k140',
       'cv19k111', 'cv19k142', 'cv19k143', 'cv19k144', 'cv19k145', 'cv19k146',
       'cr19l134', 'cs19l079', 'cs19l105', 'cs19l436', 'cs19l435', 'cf19l131',
       'cf19l132', 'cf19l133', 'cf19l134', 'cf19l135', 'cf19l136', 'cf19l504',
       'cf19l505', 'cf19l506', 'cf19l508', 'cf19l011', 'cf19l068', 'cf19l252']
    
    nomem_encr = df["nomem_encr"]
    
    df = df.loc[:, keepcols]
    
    # Load your trained model from the models directory
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
    model = load(model_path)

    # Use your trained model for prediction
    predictions = model.predict(df)
    # Return the result as a Pandas DataFrame with the columns "nomem_encr" and "prediction"
    return pd.concat([nomem_encr, pd.Series(predictions, name="prediction")], axis=1)


def predict(input_path, output):
    if output is None:
        output = sys.stdout
    df = pd.read_csv(input_path, encoding="latin-1", encoding_errors="replace", low_memory=False)
    predictions = predict_outcomes(df)
    assert (
        predictions.shape[1] == 2
    ), "Predictions must have two columns: nomem_encr and prediction"
    # Check for the columns, order does not matter
    assert set(predictions.columns) == set(
        ["nomem_encr", "prediction"]
    ), "Predictions must have two columns: nomem_encr and prediction"

    predictions.to_csv(output, index=False)


def score(prediction_path, ground_truth_path, output):
    """Score (evaluate) the predictions and write the metrics.
    
    This function takes the path to a CSV file containing predicted outcomes and the
    path to a CSV file containing the ground truth outcomes. It calculates the overall 
    prediction accuracy, and precision, recall, and F1 score for having a child 
    and writes these scores to a new output CSV file.

    This function should not be modified.
    """

    if output is None:
        output = sys.stdout
    # Load predictions and ground truth into dataframes
    predictions_df = pd.read_csv(prediction_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    # Merge predictions and ground truth on the 'id' column
    merged_df = pd.merge(predictions_df, ground_truth_df, on="nomem_encr", how="right")

    # Calculate accuracy
    accuracy = len(
        merged_df[merged_df["prediction"] == merged_df["new_child"]]
    ) / len(merged_df)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 1)]
    )
    false_positives = len(
        merged_df[(merged_df["prediction"] == 1) & (merged_df["new_child"] == 0)]
    )
    false_negatives = len(
        merged_df[(merged_df["prediction"] == 0) & (merged_df["new_child"] == 1)]
    )

    # Calculate precision, recall, and F1 score
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0
    # Write metric output to a new CSV file
    metrics_df = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_score]
    })
    metrics_df.to_csv(output, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.input_path, args.output)
    elif args.command == "score":
        score(args.prediction_path, args.ground_truth_path, args.output)
    else:
        parser.print_help()
        predict(args.input_path, args.output)  
        sys.exit(1)
