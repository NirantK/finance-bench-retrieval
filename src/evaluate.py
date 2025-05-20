from judges.classifiers.correctness import PollMultihopCorrectness
from loguru import logger


# use the correctness classifier to determine if the first model
# answered correctly
correctness = PollMultihopCorrectness(model="gpt-4o-mini")


def evaluate_rag(query: str, generated_answer: str, ground_truth: str) -> tuple[bool, str]:
    """Evaluate the RAG workflow."""
    judgment = correctness.judge(
        input=query,
        output=generated_answer,
        expected=ground_truth,
    )
    try:
        score, reasoning = judgment.score, judgment.reasoning
    except KeyError as e:
        logger.error(
            f"Error evaluating RAG: {e} with query: {query}, generated answer: {generated_answer}, ground truth: {ground_truth}"
        )
        score, reasoning = False, "Error evaluating RAG"
    return score, reasoning
