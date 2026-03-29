"""
Project B Evaluation Harness — Session 1 Starter

Multi-dimensional eval for the support pipeline:
  1. Classification accuracy — did it identify the right intent?
  2. Retrieval quality — same as Project A
  3. Response quality — faithfulness + correctness
  4. Routing accuracy — should this have been escalated to a human?

All functions are skeletons — we build them in Session 1.

Run: python scripts/eval_harness.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)


def load_golden_dataset():
    """Load Project B's golden dataset."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        print("No golden_dataset.json found for Project B. Create one first!")
        return []
    with open(path) as f:
        return json.loads(f.read())


# =========================================================================
# CLASSIFICATION METRICS
# =========================================================================

def check_classification(predicted_intent, expected_intent):
    """
    Did the system classify the query correctly?
    Returns True/False.

    TODO: Implement in Session 1.
    """
    return predicted_intent == expected_intent


# =========================================================================
# ROUTING METRICS
# =========================================================================

def check_routing(predicted_escalation, expected_escalation):
    """
    Should this query have been escalated to a human?
    Did the system make the right routing decision?

    TODO: Implement in Session 1.
    """
    return predicted_escalation == expected_escalation


# =========================================================================
# GENERATION METRICS (same pattern as Project A)
# =========================================================================

def judge_faithfulness(query, answer, context):
    """
    Is the answer grounded in the retrieved context?
    Uses GPT-4o-mini as a judge with a structured rubric.
    Returns: {"score": 1-5, "reason": "explanation"}
    
    Args:
        query: The user's question
        answer: The generated answer from the RAG system
        context: The retrieved context used to generate the answer
    
    Returns:
        Dict with 'score' (1-5) and 'reason' (explanation)
    """
    prompt = f"""You are evaluating the faithfulness of an AI assistant's answer to a question based on the provided context.

Question: {query}

Context:
{context}

Answer: {answer}

Evaluate whether the answer is faithful to the context using this rubric:

5 - Perfectly faithful: All claims in the answer are directly supported by the context
4 - Mostly faithful: Most claims are supported, minor details may be reasonable inferences
3 - Partially faithful: Some claims are supported, but there are unsupported statements
2 - Minimally faithful: Few claims are supported, significant unsupported content
1 - Not faithful: Answer contradicts context or invents information not present

Respond in JSON format:
{{"score": <1-5>, "reason": "<brief explanation>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return {"score": result["score"], "reason": result["reason"]}


def judge_correctness(query, answer, expected_answer):
    """
    Does the answer match the expected answer?
    Uses GPT-4o-mini as a judge.
    Returns: {"score": 1-5, "reason": "explanation"}
    
    Args:
        query: The user's question
        answer: The generated answer from the RAG system
        expected_answer: The ground truth answer from the golden dataset
    
    Returns:
        Dict with 'score' (1-5) and 'reason' (explanation)
    """
    prompt = f"""You are evaluating the correctness of an AI assistant's answer compared to a reference answer.

Question: {query}

Reference Answer (Ground Truth): {expected_answer}

Generated Answer: {answer}

Evaluate how well the generated answer matches the reference answer using this rubric:

5 - Perfect: Answers are semantically equivalent, all key information matches
4 - Good: Most key information matches, minor differences in wording or detail
3 - Acceptable: Main point is correct, but missing some important details
2 - Poor: Partially correct but has significant errors or omissions
1 - Wrong: Answer is incorrect or contradicts the reference answer

Respond in JSON format:
{{"score": <1-5>, "reason": "<brief explanation>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return {"score": result["score"], "reason": result["reason"]}


# =========================================================================
# EVAL RUNNER
# =========================================================================

def run_eval():
    """
    Run multi-dimensional eval:
    1. Classification accuracy
    2. Retrieval quality
    3. Response quality (faithfulness + correctness)
    4. Routing accuracy

    TODO: Implement in Session 1.
    """
    from support_pipeline import handle_query
    from retrieval import embed_query, retrieve, assemble_context
    
    golden_data = load_golden_dataset()
    if not golden_data:
        print("No golden dataset found. Exiting.")
        return
    
    print(f"\n{'='*70}")
    print(f"Running 4-Dimensional Evaluation on {len(golden_data)} queries")
    print(f"{'='*70}\n")
    
    # Metrics accumulators
    classification_correct = 0
    faithfulness_scores = []
    correctness_scores = []
    routing_correct = 0
    correct_handle = 0  # Should NOT escalate, didn't escalate
    missed_escalate = 0  # Should escalate, didn't escalate
    total_should_not_escalate = 0
    total_should_escalate = 0
    
    for i, entry in enumerate(golden_data, 1):
        query = entry["query"]
        expected_intent = entry.get("expected_intent", "")
        expected_answer = entry.get("expected_answer", "")
        expected_escalation = entry.get("expected_escalation", False)
        
        print(f"\n[{i}/{len(golden_data)}] Processing: {query[:60]}...")
        
        # Run the pipeline
        result = handle_query(query)
        predicted_intent = result["intent"]
        answer = result["answer"]
        
        # Naive pipeline never escalates
        predicted_escalation = False
        
        # Get retrieval context for faithfulness check
        query_embedding = embed_query(query)
        chunks = retrieve(query_embedding)
        context = assemble_context(chunks)
        
        # 1. Classification accuracy
        classification_pass = check_classification(predicted_intent, expected_intent)
        if classification_pass:
            classification_correct += 1
        
        # 2. Faithfulness
        faithfulness_result = judge_faithfulness(query, answer, context)
        faithfulness_scores.append(faithfulness_result["score"])
        
        # 3. Correctness
        correctness_result = judge_correctness(query, answer, expected_answer)
        correctness_scores.append(correctness_result["score"])
        
        # 4. Routing accuracy
        routing_pass = check_routing(predicted_escalation, expected_escalation)
        if routing_pass:
            routing_correct += 1
        
        # Track routing breakdown
        if expected_escalation:
            total_should_escalate += 1
            if not predicted_escalation:
                missed_escalate += 1
        else:
            total_should_not_escalate += 1
            if not predicted_escalation:
                correct_handle += 1
    
    # Calculate overall metrics
    total = len(golden_data)
    classification_acc = (classification_correct / total) * 100
    avg_faithfulness = sum(faithfulness_scores) / total
    avg_correctness = sum(correctness_scores) / total
    routing_acc = (routing_correct / total) * 100
    
    # Routing breakdown percentages
    correct_handle_pct = (correct_handle / total_should_not_escalate * 100) if total_should_not_escalate > 0 else 0
    missed_escalate_pct = (missed_escalate / total_should_escalate * 100) if total_should_escalate > 0 else 0
    
    # Calculate missed escalation caught rate (inverse of missed escalate %)
    escalation_caught_pct = 100 - missed_escalate_pct if total_should_escalate > 0 else 0
    
    # Print scorecard
    print(f"\n{'='*70}")
    print(f"EVALUATION SCORECARD")
    print(f"{'='*70}\n")
    
    print(f"1. CLASSIFICATION ACCURACY: {classification_acc:.1f}%")
    print(f"   ({classification_correct}/{total} correct)\n")
    
    print(f"2. RESPONSE FAITHFULNESS: {avg_faithfulness:.2f}/5.0")
    print(f"   (grounded in context)\n")
    
    print(f"3. RESPONSE CORRECTNESS: {avg_correctness:.2f}/5.0")
    print(f"   (matches expected answer)\n")
    
    print(f"4. ROUTING ACCURACY: {routing_acc:.1f}%")
    print(f"   Overall: {routing_correct}/{total} correct")
    print(f"   Routing breakdown:")
    print(f"     - {correct_handle_pct:.1f}% correct-handle (should NOT escalate)")
    print(f"     - {escalation_caught_pct:.1f}% missed-escalation caught (should escalate)")
    
    print(f"\n{'='*70}\n")


def run_stratified_eval():
    """
    Run stratified evaluation by intent.
    Shows per-intent classification accuracy and confusion patterns.
    Helps identify which intents the classifier struggles with.
    """
    from support_pipeline import handle_query
    from collections import defaultdict
    
    golden_data = load_golden_dataset()
    if not golden_data:
        print("No golden dataset found. Exiting.")
        return
    
    print(f"\n{'='*70}")
    print(f"STRATIFIED EVALUATION BY INTENT")
    print(f"{'='*70}\n")
    
    # Group entries by expected intent
    intent_groups = defaultdict(list)
    for entry in golden_data:
        intent_groups[entry["expected_intent"]].append(entry)
    
    # Track metrics per intent
    intent_metrics = {}
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # Process each intent group
    for expected_intent, entries in sorted(intent_groups.items()):
        print(f"\nProcessing {expected_intent}: {len(entries)} queries...")
        
        correct = 0
        total = len(entries)
        
        for entry in entries:
            query = entry["query"]
            
            # Run classification
            result = handle_query(query)
            predicted_intent = result["intent"]
            
            # Track results
            if predicted_intent == expected_intent:
                correct += 1
            
            # Update confusion matrix
            confusion_matrix[expected_intent][predicted_intent] += 1
        
        # Calculate accuracy for this intent
        accuracy = (correct / total) * 100 if total > 0 else 0
        intent_metrics[expected_intent] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
    
    # Print per-intent results
    print(f"\n{'='*70}")
    print(f"PER-INTENT CLASSIFICATION ACCURACY")
    print(f"{'='*70}\n")
    
    for intent in sorted(intent_metrics.keys()):
        metrics = intent_metrics[intent]
        print(f"{intent}:")
        print(f"  Accuracy: {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total']} correct)")
    
    # Overall accuracy
    total_correct = sum(m["correct"] for m in intent_metrics.values())
    total_queries = sum(m["total"] for m in intent_metrics.values())
    overall_acc = (total_correct / total_queries) * 100 if total_queries > 0 else 0
    
    print(f"\nOVERALL CLASSIFICATION ACCURACY: {overall_acc:.1f}%")
    print(f"({total_correct}/{total_queries} correct)")
    
    # Print confusion matrix
    print(f"\n{'='*70}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*70}\n")
    print("Shows: Expected Intent → Predicted Intent (count)")
    print()
    
    for expected in sorted(confusion_matrix.keys()):
        print(f"\n{expected}:")
        predictions = confusion_matrix[expected]
        
        # Sort by count (descending)
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for predicted, count in sorted_predictions:
            if predicted == expected:
                print(f"  ✓ {predicted}: {count} (CORRECT)")
            else:
                print(f"  ✗ {predicted}: {count} (misclassified)")
    
    # Identify common misclassification patterns
    print(f"\n{'='*70}")
    print(f"COMMON MISCLASSIFICATION PATTERNS")
    print(f"{'='*70}\n")
    
    misclassifications = []
    for expected in confusion_matrix:
        for predicted, count in confusion_matrix[expected].items():
            if expected != predicted:
                misclassifications.append((count, expected, predicted))
    
    misclassifications.sort(reverse=True)
    
    if misclassifications:
        print("Top confusion pairs (count, expected → predicted):\n")
        for count, expected, predicted in misclassifications[:10]:
            print(f"  {count}x: {expected} → {predicted}")
    else:
        print("No misclassifications found! Perfect classification!")
    
    # Check for "general" fallback pattern
    print(f"\n{'='*70}")
    print(f"FALLBACK TO 'general' ANALYSIS")
    print(f"{'='*70}\n")
    
    general_fallback_count = 0
    total_misclassified = 0
    
    for expected in confusion_matrix:
        if expected != "general":
            general_count = confusion_matrix[expected].get("general", 0)
            general_fallback_count += general_count
        
        for predicted, count in confusion_matrix[expected].items():
            if expected != predicted:
                total_misclassified += count
    
    if total_misclassified > 0:
        fallback_pct = (general_fallback_count / total_misclassified) * 100
        print(f"Total misclassifications: {total_misclassified}")
        print(f"Misclassified as 'general': {general_fallback_count} ({fallback_pct:.1f}%)")
        
        if fallback_pct > 30:
            print("\n⚠️  WARNING: Classifier is falling back to 'general' frequently!")
        elif fallback_pct > 0:
            print(f"\nClassifier occasionally falls back to 'general'")
        else:
            print(f"\nNo 'general' fallback detected")
    else:
        print("No misclassifications found!")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    print("Project B Evaluation Harness")
    print("Multi-dimensional eval: classification + retrieval + response + routing")
    print()
    
    # Check if user wants stratified eval
    if len(sys.argv) > 1 and sys.argv[1] == "--stratified":
        run_stratified_eval()
    elif len(sys.argv) > 1 and sys.argv[1] == "--both":
        run_eval()
        run_stratified_eval()
    else:
        print("Usage:")
        print("  python scripts/eval_harness.py           # Run full 4-dimensional eval")
        print("  python scripts/eval_harness.py --stratified  # Run stratified eval by intent")
        print("  python scripts/eval_harness.py --both       # Run both evaluations")
        print()
        run_eval()
