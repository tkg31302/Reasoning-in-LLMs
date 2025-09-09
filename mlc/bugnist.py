"""BugNIST model and scoring template."""

import re
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
from skimage.io import imread
from pathlib import Path
DATA = Path(__file__).resolve().parent / "test_data"
class ScorableModelTemplate(ABC):
    """Abstract class for scorable models.

    This is a template class meant to be inherited. Sub-classes
    should implement .process_inputs and .predict. A test case
    is ran when a sub-class is initialize to help debug common
    issues.
    """

    def __new__(cls, *args, **kwargs):
        """Checks the subclass has been implmented correctly.

        Allows overwriting __init__ in subclasses.
        """
        instance = super().__new__(cls)
        instance.__check_rep__()
        return instance

    def load_test_case(self):
        """Load test case to check that the model works

        :return raw_input: 3d np array of volume
        :return true_output: pd.DataFrame of true target prediction
        """
        raw_input = imread(str(DATA / "mix_02_006.tif"))
        true_output = pd.read_csv(str(DATA / "mix_02_006.csv"))
        return raw_input, true_output

    def __check_rep__(self):
        """Return predicted values."""

        # Read in test data
        _, true_output = self.load_test_case()

        # Predict
        try:
            predicted_output = self.predict([str(DATA / "mix_02_006.tif")])
        except Exception as e:
            raise ValueError(f"predict function does not work: {e}")

        # Score
        try:
            compute_score(true_output, predicted_output)
        except Exception as e:
            raise ValueError(f"prediction could not be scored: {e}")

        # Checks
        assert isinstance(predicted_output, pd.DataFrame), "output should be a pd.DataFrame"
        assert len(predicted_output.columns) == len(true_output.columns), "there should be two columns returned"
        assert np.all(predicted_output.columns == true_output.columns), "predicted dataframe columns should be: ['filename', 'centerpoints']"
        assert len(predicted_output) == 1, f"test case only has one example but {len(predicted_output)} were predicted"

    # Implement these functions:
    @abstractmethod
    def predict(self, raw_files):
        """Predict labels and positions of bugs.

        :param raw_files: list of file path strings
        :return predictions: dataframe with columns: ["filename", "centerpoints"]
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, raw_files):
        """Read in arrays and compute features.

        :param raw_files: list of file path strings
        :return: anything needed for you model to make predictions, e.g. features or processed data
        """
        raise NotImplementedError()


# Scoring function from Kaggle:
def compute_score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Bugnist scoring.

    Calculate the score for the given solution and submission dataframes.

    Args:
        solution (pd.DataFrame): DataFrame containing the ground truth data.
        submission (pd.DataFrame): DataFrame containing the submission data.

    Returns:
        float: Score for the submission.
    """
    assert isinstance(solution, pd.DataFrame)
    assert isinstance(submission, pd.DataFrame)

    f1_scores = []
    for i in range(len(submission)):
        if pd.isna(submission.iloc[i, 1]):
            raise ParticipantVisibleError(f"Nan value not supported. Found on line {i} in submission file")

        # Extracts data for each line
        pred_centerpoints = submission.iloc[i, 1].replace(" ", "").rstrip(";").split(";")  # Removes whitespace and last ';' if applicable
        sol_centerpoints = solution.iloc[i, 1].split(";")
        pred_filename = str(submission.iloc[i, 0])
        sol_filename = str(solution.iloc[i, 0])

        if pred_filename != sol_filename:
            raise ParticipantVisibleError("Internal error: solution and submission are not lined up")

        if len(pred_centerpoints) % 4 != 0:
            raise ParticipantVisibleError(
                f"Submission for file {pred_filename}, index {i} could not be separated based on ';' into segmentations of size 4. Instead, got a list of size {len(pred_centerpoints)} % 4 != 0"
            )

        # Extract center coordinates and labels from submission and solution
        filtered_centerpoints = [float(item) for item in pred_centerpoints if not re.search("[a-zA-Z]", item)]
        filtered_true_centerpoints = [float(item) for item in sol_centerpoints if not re.search("[a-zA-Z]", item)]
        pred_centers = np.array(list(zip(filtered_centerpoints[::3], filtered_centerpoints[1::3], filtered_centerpoints[2::3])))
        true_centers = np.array(list(zip(filtered_true_centerpoints[::3], filtered_true_centerpoints[1::3], filtered_true_centerpoints[2::3])))

        # Converts labels to numbers
        index_to_label = np.array(["sl", "bc", "ma", "gh", "ac", "bp", "bf", "cf", "bl", "ml", "wo", "pp"])
        label_to_index = {k.lower(): i for i, k in enumerate(index_to_label)}
        label_to_index["gp"] = label_to_index["bp"]

        try:
            pred_labels = np.array([label_to_index[label.lower()] for label in pred_centerpoints[::4]])
            true_labels = np.array([label_to_index[label.lower()] for label in sol_centerpoints[::4]])
        except KeyError:
            raise ParticipantVisibleError(f"Invalid class label in {pred_centerpoints[::4]}, should match any of {index_to_label}")

        # Calculate cost matrix and perform matching
        cost = cdist(pred_centers, true_centers)
        matches = np.array(scipy.optimize.linear_sum_assignment(cost), dtype=np.int32)

        # Filter matches based on matched labels
        matched_box_labels = pred_labels[matches[0]]
        matched_center_labels = true_labels[matches[1]]
        matches = matches[:, matched_box_labels == matched_center_labels]

        # Calculate F1 score
        f1_score = match_precision_recall(matches, pred_labels, true_labels)[3]
        f1_scores.append(f1_score)

    return np.mean(f1_scores)

class ParticipantVisibleError(Exception):
    """Custom exception class for errors that should be shown to participants."""
    pass

def match_precision_recall(matches: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray, eps: float = 1e-6) -> tuple:
    """
    Calculate precision and recall for detection and class matching.

    Args:
        matches: (2, K) array of box to center matches
        pred_labels: (N,) array of predicted labels
        true_labels: (M,) array of true labels
        eps: Small value to avoid division by zero

    Returns:
        Tuple containing precision and recall for detection and class matching
    """
    pred_match_labels = pred_labels[matches[0]]
    true_match_labels = true_labels[matches[1]]
    matches_class = matches[:, pred_match_labels == true_match_labels]

    # Detection metrics
    precision_detect = matches.shape[1] / pred_labels.shape[0]
    recall_detect = matches.shape[1] / true_labels.shape[0]
    f1_detect = 2 * precision_detect * recall_detect / (precision_detect + recall_detect + eps)

    # Class metrics
    precision_classes = matches_class.shape[1] / pred_labels.shape[0]
    recall_classes = matches_class.shape[1] / true_labels.shape[0]
    f1_classes = 2 * precision_classes * recall_classes / (precision_classes + recall_classes + eps)

    return f1_detect, precision_detect, recall_detect, f1_classes, precision_classes, recall_classes