import unittest

from app.app import RAG

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,  # Do not worry about the warning: https://github.com/explodinggradients/ragas/issues/993
)


class BasicRagTest(unittest.TestCase):
    """
    The BasicRagTest class is designed to perform unit tests on the functionality of the Rag class,
    ensuring that the dataset generated from the Rag output is correctly evaluated against specific
    metrics. Each test focuses on a particular aspect of the evaluation, such as the presence of
    expected entries in the dataset and the adherence of various metrics (context utilization, faithfulness,
    and answer relevancy) to predefined thresholds.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up method to initialize common test data and procedures. This method is called once before
        executing all tests.
        """
        rag_ = RAG()
        data_json = rag_.main()
        cls.dataset = Dataset.from_dict(data_json)
        cls.result = evaluate(dataset=cls.dataset,
                              metrics=[
                                  context_utilization,
                                  faithfulness,
                                  answer_relevancy,
                              ])
        cls.df = cls.result.to_pandas()

    def test_dataset_entries(self):
        """
        Test to ensure the dataset contains the expected number of entries.
        """
        expected_entries = 3
        self.assertEqual(expected_entries, self.df.shape[0], f"Expected {expected_entries} entries in the dataset, found {self.df.shape[0]}.")

    def test_faithfulness_score(self):
        """
        Test to ensure the average faithfulness score is above a certain threshold, indicating
        that the responses are generally faithful to the given context.
        """
        min_faithfulness = 0.5
        mean_faithfulness = self.df.faithfulness.mean()
        self.assertGreater(mean_faithfulness, min_faithfulness, f"Average faithfulness ({mean_faithfulness}) is below the minimum threshold of {min_faithfulness}.")

    def test_context_utilization_score(self):
        """
        Test to ensure the average context utilization score is not negative, which would indicate
        a relevance of the generated content to the provided context.
        """
        self.assertGreaterEqual(self.df.context_utilization.mean(), 0, "The average context utilization is negative, indicating possible irrelevance of content to context.")

    def test_answer_relevancy_score(self):
        """
        Test to ensure the average answer relevancy score is above a certain threshold, affirming
        that the generated answers are relevant to the questions posed.
        """
        min_relevancy = 0.5
        mean_relevancy = self.df.answer_relevancy.mean()
        self.assertGreater(mean_relevancy, min_relevancy, f"Average answer relevancy ({mean_relevancy}) is below the minimum threshold of {min_relevancy}.")
