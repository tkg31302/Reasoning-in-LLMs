"""BirdCLEF model and scoring template."""
from pathlib import Path
import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics
import soundfile as sf
from abc import ABC, abstractmethod
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

        :return raw_input: np.array audio signal
        :return true_output: pd.DataFrame of true target prediction
        """
        raw_input, _ = sf.read(str(DATA / "H02_20230420_074000.ogg"))
        true_output = pd.read_csv(str(DATA / "H02_20230420_074000.csv"))
        return raw_input, true_output

    def __check_rep__(self):
        """Return predicted values."""

        # Read in test data
        _, true_output = self.load_test_case()

        # Predict
        try:
            predicted_output = self.predict([str(DATA / "H02_20230420_074000.ogg")])
        except Exception as e:
            raise ValueError(f"predict function does not work: {e}")

        # Score
        try:
            compute_score(true_output, predicted_output)
        except Exception as e:
            raise ValueError(f"prediction could not be scored: {e}")

        # Checks
        assert isinstance(predicted_output, pd.DataFrame), "output should be a pd.DataFrame"
        assert len(predicted_output) == len(true_output), "every 5s segment should get a prediction"
        assert len(predicted_output.columns) == len(true_output.columns), "there should be 207 columns returned"
        assert np.all(predicted_output.values[:, 1:] >= 0) and np.all(predicted_output.values[:, 1:] <= 1), "predictions should be a float between 0 and 1"

    # Implement these function:
    @abstractmethod
    def predict(self, raw_files: list[str]):
        """Predict labels and positions of bugs.

        :param raw_files: list of file path strings
        :return predictions: dataframe with columns: "row_id" and a each of the 206 class names
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, raw_files: list[str]):
        """Read in arrays and compute features.

        :param raw_files: list of file path strings
        :return: anything needed for you model to make predictions, e.g. features or processed data
        """
        raise NotImplementedError()


# Scoring function from Kaggle:
class ParticipantVisibleError(Exception):
    pass

def compute_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name="row_id") -> float:
    """Macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.

    :param solution: solutions
    :param submission: predictions
    :param row_id_column_name: name of column, should be "row_id", to drop when scoring
    :return score: auc score
    """
    assert isinstance(solution, pd.DataFrame)
    assert isinstance(submission, pd.DataFrame)

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype  for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return sklearn.metrics.roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average='macro')

def get_class_names():
    """Get class names from the header of the solution.

    :return: list of class names
    """
    return pd.read_csv("mlc/test_data/H02_20230420_074000.csv").columns[1:].tolist()