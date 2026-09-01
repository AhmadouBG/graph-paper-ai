# test_vectorless_rag.py
import pytest
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase
from deepeval.conversation_simulator import ConversationSimulator
from deepeval.dataset import EvaluationDataset
from deepeval.models import AmazonBedrockModel
from deepeval.metrics import (
    TurnFaithfulnessMetric,
    TurnContextualRelevancyMetric,
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
)
from your_app import model_callback

# Bedrock as LLM judge for all metrics
bedrock_judge = AmazonBedrockModel(
    model="anthropic.claude-3-opus-20240229-v1:0",
    region="us-east-1",
    generation_kwargs={"temperature": 0},
)

# Pull pre-generated goldens
dataset = EvaluationDataset()
dataset.pull(alias="Vectorless RAG Dataset")

# Simulate real conversations using your pipeline
simulator = ConversationSimulator(model_callback=model_callback)
test_cases = simulator.simulate(
    conversational_goldens=dataset.goldens,
    max_user_simulations=10,
)

# Define metrics — all using Bedrock as judge
metrics = [
    TurnFaithfulnessMetric(threshold=0.7, model=bedrock_judge),        # answer grounded in tree nodes?
    TurnContextualRelevancyMetric(threshold=0.7, model=bedrock_judge), # retrieved sections relevant?
    TurnContextualPrecisionMetric(threshold=0.7, model=bedrock_judge), # best nodes ranked first?
    TurnContextualRecallMetric(threshold=0.7, model=bedrock_judge),    # all relevant nodes retrieved?
]

@pytest.mark.parametrize("test_case", test_cases)
def test_vectorless_rag(test_case: ConversationalTestCase):
    assert_test(test_case=test_case, metrics=metrics)