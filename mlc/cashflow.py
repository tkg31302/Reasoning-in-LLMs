"""Cashflow model and scoring template."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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

        :return raw_input: dataframe of transactions for one person
        :return true_output: pd.DataFrame of true target prediction
        """
        raw_input = pd.read_parquet(str(DATA / "transactions.parquet"))
        true_output = pd.read_parquet(str(DATA / "consumer_data.parquet"))
        return raw_input, true_output

    def __check_rep__(self):
        """Return predicted values."""

        # Read in test data
        _, true_output = self.load_test_case()

        # Predict
        try:
            predicted_output = self.predict(str(DATA / "consumer_data.parquet"), str(DATA / "transactions.parquet"))
        except Exception as e:
            raise ValueError(f"predict function does not work: {e}")

        # Score
        try:
            compute_score(true_output, predicted_output)
        except Exception as e:
            raise ValueError(f"prediction could not be scored: {e}")

        # Checks
        accepted_types = (np.ndarray, list, tuple, pd.Series)
        assert isinstance(predicted_output, accepted_types), "output should be scorable by roc_auc_score, e.g. np.array"
        assert len(predicted_output) == len(true_output), "each row in consumer_data needs a prediction"
        assert np.all(predicted_output >= 0) and np.all(predicted_output <= 1), "predictions should be a float between 0 and 1"

    # Implement these function:
    @abstractmethod
    def predict(self, raw_consumer_file: str, raw_transactions_file: str):
        """Predict labels and positions of bugs.

        :param raw_consumer_file: path to consumer_data.parquet
        :param raw_consumer_file: path to transactions.parquet
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, raw_consumer_file: str, raw_transactions_file: str):
        """Read in arrays and compute features.

        :param raw_files: path to transactions.parquet
        :return: anything needed for you model to make predictions, e.g. features or processed data
        """
        raise NotImplementedError()

def compute_score(df_consumer: pd.DataFrame, y_pred: np.ndarray):
    """Compute scores for cashflow.

    :param df_consumer: dataframe loaded from consumer_data.parquet
    :param y_pred: prediction for each row in df_consumer
    :return score: min auc across consumer groups
    """
    assert isinstance(df_consumer, pd.DataFrame)
    assert isinstance(y_pred, (np.ndarray, list, tuple, pd.Series))

    df_consumer['group_id'] = df_consumer['masked_consumer_id'].str[:3]
    df_consumer['y_pred'] = y_pred
    return min([roc_auc_score(df["FPF_TARGET"], df["y_pred"]) for _, df in df_consumer.groupby('group_id')])