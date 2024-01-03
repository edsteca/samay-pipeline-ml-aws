"""Evaluation script for measuring model accuracy."""
import joblib
import json
import logging
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd
import xgboost

from sklearn import model_selection
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = joblib.load("model.joblib")

    test_path = "/opt/ml/processing/output/data_to_train.csv"
    df = pd.read_csv(test_path, index_col=0,sep=';')
    print(df.head(5))

    sound_df = df
    X = np.array(data.drop(['Fracaso_o_exito'],1))
    y = np.array(data['Fracaso_o_exito'])
    
    print('Shape : ', X.shape)

    # Initialize the LabelEncoder
    le = LabelEncoder()
    
    # Fit the label encoder to the 'label' column and return the encoded labels
    encoded_labels = le.fit_transform(sound_df['label'])

    # Split the scaled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_data, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
    )
  
    predictions = model.predict(X_test)
    
    logger.debug("Reading test data.")

    logger.info("Performing predictions against test data.")


    precision = precision_score(y_test, predictions, average="binary", pos_label="Éxito")#check
    recall = recall_score(y_test, predictions, average="binary", pos_label="Éxito")#check
    accuracy = accuracy_score(y_test, predictions)  #check
    conf_matrix = confusion_matrix(y_test, predictions) #check
    fpr, tpr, _ = roc_curve(y_testb, predictions_b)

    logger.debug("Accuracy: {}".format(accuracy))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(conf_matrix))

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "confusion_matrix": conf_matrix ,
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
