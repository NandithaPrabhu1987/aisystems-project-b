# Week 1: Evaluation & Baseline Results

**AI Systems in Production — Project B**  
**Session 1: Multi-Dimensional Evaluation Framework**

## Overview

Week 1 focuses on building a comprehensive evaluation harness to measure the naive support pipeline across 4 dimensions:
1. **Classification Accuracy** — Did it identify the right intent?
2. **Response Faithfulness** — Is the answer grounded in retrieved context?
3. **Response Correctness** — Does it match the expected answer?
4. **Routing Accuracy** — Should this have been escalated to a human?

## Golden Dataset

Created `scripts/golden_dataset.json` with **30 test queries**:

- **6 intents covered**: return_or_refund, order_status, billing_or_payment, product_info, membership, general
- **At least 4 entries per intent** for balanced evaluation
- **9 escalation cases** (30% of dataset) requiring human intervention:
  - Billing disputes (duplicate charges, unauthorized charges)
  - Account security breaches
  - Damaged deliveries
  - Complex edge cases (multi-policy conflicts, technical blocking issues)
- **21 standard queries** for automated handling:
  - Straightforward policy lookups
  - Standard product questions
  - Simple returns and inquiries

### Dataset Fields

Each entry contains:
```json
{
  "id": 1,
  "query": "Customer query text",
  "expected_intent": "return_or_refund",
  "expected_answer": "Ground truth answer",
  "expected_source": "01_return_policy.md",
  "expected_escalation": false,
  "difficulty": "easy",
  "category": "policy_lookup"
}
```

## Evaluation Implementation

### Functions Implemented

1. **`check_classification(predicted, expected)`**
   - Simple string equality for intent classification
   - Returns True/False
   - Surfaces which intents classifier gets wrong most consistently

2. **`check_routing(predicted_escalation, expected_escalation)`**
   - Boolean equality for escalation decisions
   - Returns True/False
   - Baseline: naive pipeline always returns False (never escalates)

3. **`judge_faithfulness(query, answer, context)`**
   - LLM-as-judge using GPT-4o-mini
   - 5-point rubric for grounding in context
   - Returns: `{"score": 1-5, "reason": "explanation"}`

4. **`judge_correctness(query, answer, expected_answer)`**
   - LLM-as-judge using GPT-4o-mini
   - 5-point rubric for semantic equivalence
   - Returns: `{"score": 1-5, "reason": "explanation"}`

5. **`run_eval()`**
   - Full 4-dimensional evaluation
   - Processes all 30 queries through the pipeline
   - Generates comprehensive scorecard

6. **`run_stratified_eval()`**
   - Per-intent classification breakdown
   - Confusion matrix analysis
   - Identifies misclassification patterns
   - Detects 'general' fallback behavior

## Running the Evaluation

```bash
# Full 4-dimensional evaluation
python scripts/eval_harness.py

# Stratified evaluation (classification by intent)
python scripts/eval_harness.py --stratified

# Both evaluations
python scripts/eval_harness.py --both
```

## Baseline Results (Naive Pipeline)

### Overall Metrics

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Classification Accuracy** | **90.0%** (27/30) | Strong baseline classification |
| **Response Faithfulness** | **4.70/5.0** | Answers well-grounded in retrieved context |
| **Response Correctness** | **3.00/5.0** | Semantic match with expected answers |
| **Routing Accuracy** | **66.7%** (20/30) | Never escalates (0% escalation detection) |

### Routing Breakdown

- **Correct-Handle**: 100.0% (21/21 cases that shouldn't escalate)
- **Missed-Escalation Caught**: 0.0% (0/9 cases that should escalate)

**Critical Finding**: The naive pipeline never escalates, resulting in 0% detection of cases requiring human intervention.

### Per-Intent Classification Accuracy

| Intent | Accuracy | Correct/Total | Notes |
|--------|----------|---------------|-------|
| **billing_or_payment** | 100.0% | 5/5 | ✅ Perfect |
| **membership** | 100.0% | 5/5 | ✅ Perfect |
| **product_info** | 100.0% | 4/4 | ✅ Perfect |
| **return_or_refund** | 83.3% | 5/6 | 1 misclassified as product_info |
| **order_status** | 80.0% | 4/5 | 1 misclassified as general |
| **general** | 80.0% | 4/5 | 1 misclassified as membership |

**Overall Classification**: 90.0% (27/30 correct)

### Confusion Matrix

```
billing_or_payment:
  ✓ billing_or_payment: 5 (CORRECT)

general:
  ✓ general: 4 (CORRECT)
  ✗ membership: 1 (misclassified)

membership:
  ✓ membership: 5 (CORRECT)

order_status:
  ✓ order_status: 4 (CORRECT)
  ✗ general: 1 (misclassified)

product_info:
  ✓ product_info: 4 (CORRECT)

return_or_refund:
  ✓ return_or_refund: 5 (CORRECT)
  ✗ product_info: 1 (misclassified)
```

### Common Misclassification Patterns

1. **return_or_refund → product_info** (1x)
   - Query about warranty/product defects confused with product inquiry
   
2. **order_status → general** (1x)
   - Complex order issue misclassified as general inquiry
   
3. **general → membership** (1x)
   - Account-related general query confused with membership

### Fallback Analysis

- **Total misclassifications**: 3
- **Misclassified as 'general'**: 1 (33.3% of errors)
- ⚠️ **WARNING**: Classifier is falling back to 'general' frequently!

## Key Findings

### ✅ Strengths

1. **High Classification Accuracy (90%)**
   - Perfect accuracy on billing, membership, and product queries
   - Only 3 misclassifications out of 30 queries

2. **Excellent Faithfulness (4.70/5.0)**
   - RAG retrieval working well
   - Answers consistently grounded in policy documents
   - Minimal hallucination

3. **Strong Intent Coverage**
   - System handles 6 different intent types
   - Good distribution across use cases

### ⚠️ Areas for Improvement

1. **Zero Escalation Detection (0%)**
   - **CRITICAL**: Naive pipeline never escalates
   - Misses all 9 cases requiring human intervention
   - Billing disputes, security issues, damaged deliveries all auto-handled
   - Risk of real harm in production

2. **Moderate Correctness (3.00/5.0)**
   - Answers are faithful but not always complete
   - Missing some details from expected answers
   - Semantic match could be improved

3. **General Intent Fallback**
   - 33% of errors fall back to 'general'
   - Indicates classifier uncertainty on edge cases

4. **Intent Confusion Patterns**
   - return_or_refund ↔ product_info overlap
   - order_status ↔ general boundary unclear
   - Need better few-shot examples or prompt engineering

## Escalation Cases Analysis

**9 queries require escalation but all were auto-handled:**

| ID | Query Type | Reason for Escalation | Risk if Not Escalated |
|----|-----------|----------------------|----------------------|
| 2 | Damaged laptop delivery | Shipping damage investigation | Customer dissatisfaction, bad review |
| 4 | Return without packaging + warranty claim | Multi-policy edge case | Incorrect policy application |
| 6 | Missing high-value package (₹50,000) | Security/theft investigation | Financial loss, fraud |
| 10 | Duplicate charge | Billing dispute | Payment issue, trust damage |
| 12 | Payment debited, no order | Payment reconciliation | Financial loss, legal issue |
| 16 | Professional video editing requirements | Expert consultation needed | Wrong product recommendation |
| 19 | Unauthorized auto-renewal | Billing dispute | Unauthorized charge |
| 23 | Account security breach | Security incident | Data breach, fraud |
| 26 | Misdelivery + unresponsive support | Service failure escalation | Lost package, poor experience |
| 29 | Technical issue blocking return | System failure near deadline | Customer unable to return, harm |

**Impact**: Without escalation logic, these 9 critical cases would be handled incorrectly in production.

## Next Steps for Week 2+

### Priority 1: Implement Escalation Logic
- [ ] Add confidence scoring to responses
- [ ] Implement escalation decision tree
- [ ] Test against golden dataset escalation cases
- [ ] Target: >80% escalation detection rate

### Priority 2: Improve Classification
- [ ] Add few-shot examples for confused intents
- [ ] Better prompt engineering for return_or_refund vs product_info
- [ ] Reduce 'general' fallback rate
- [ ] Target: >95% classification accuracy

### Priority 3: Enhance Response Quality
- [ ] Improve completeness of answers
- [ ] Add structured citations
- [ ] Better handling of multi-policy questions
- [ ] Target: >4.0/5.0 correctness score

### Priority 4: Add Tool Use (Week 2)
- [ ] Customer record lookup tool
- [ ] Order status tracker tool
- [ ] Query classifier for tool selection
- [ ] Proto-reasoning layer

### Priority 5: Build LangGraph Agent (Week 3)
- [ ] Multi-step planning
- [ ] Agent decision tracing
- [ ] Model routing (GPT-4 for complex, GPT-4o-mini for simple)
- [ ] Full observability with LangFuse

## Metrics to Track Weekly

| Week | Classification | Faithfulness | Correctness | Routing | Escalation Detection |
|------|---------------|--------------|-------------|---------|---------------------|
| 1 (Baseline) | 90.0% | 4.70/5.0 | 3.00/5.0 | 66.7% | 0.0% |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD |

## Files Modified/Created in Week 1

- ✅ `scripts/eval_harness.py` — Full evaluation implementation
- ✅ `scripts/golden_dataset.json` — 30-query test dataset
- ✅ `.env` — Environment configuration
- ✅ `.gitignore` — Updated with venv, .env, temp files
- ✅ `WEEK1_RESULTS.md` — This document

## Conclusion

Week 1 establishes a **strong baseline** with 90% classification accuracy and excellent faithfulness (4.70/5.0). However, the **0% escalation detection rate is critical** and must be addressed in Week 2. The evaluation framework is robust and ready to measure improvements as we evolve from a naive pipeline to a full production agent.

The stratified evaluation reveals that while most intents are handled well, there's room for improvement in distinguishing edge cases and preventing 'general' fallback. With 9 escalation cases in the dataset, we have clear targets for improving routing accuracy.

**Ready for Week 2**: Tool use, query classification, and proto-reasoning layer! 🚀
