# Project B — Customer Support Agent with Tool Use & Human Escalation

Part of **AI Systems in Production** — AI Classroom Cohort 3

A LangGraph-based agent that receives customer queries, reasons about which tools to use, executes a multi-step plan, and produces structured responses with citations and confidence-based human escalation. Evolves from a naive pipeline in Week 1 into a full production agent by Week 4.

## Setup

### 1. Prerequisites
- Python 3.11+
- Docker Desktop
- OpenAI API key (with credits)
- LangFuse account (cloud.langfuse.com — free tier)

### 2. Environment

```bash
cp .env.example .env
# Fill in your API keys in .env

docker-compose up -d

pip install -r requirements.txt
```

### 3. Run

```bash
# Set up the database (retrieval over policy corpus)
python scripts/setup_db.py

# Ingest policy documents
python scripts/ingest.py

# Test the support pipeline
python scripts/support_pipeline.py

# Eval harness
python scripts/eval_harness.py
```

## Repo Structure

```
project-b/
├── corpus/                 # Acmera policy documents (19 files, same as Project A)
├── mock_data/
│   ├── customers.json      # Simulated customer DB (order history, tier, account status)
│   └── orders.json         # Simulated order status + shipping data
├── scripts/
│   ├── setup_db.py         # Create pgvector table for policy retrieval
│   ├── ingest.py           # Chunk + embed + store policy corpus
│   ├── retrieval.py        # Self-contained retrieval layer (embed + retrieve + assemble)
│   ├── support_pipeline.py # Naive pipeline: classify → retrieve → respond
│   └── eval_harness.py     # Multi-dimensional eval skeleton (built in Session 1)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## What We Build (Week by Week)

| Week | Layer | What Gets Added |
|------|-------|-----------------|
| 1 | Evaluate | Multi-dimensional eval: classification + retrieval + response + routing accuracy |
| 2 | Retrieve | Query classifier for tool selection, proto-reasoning layer |
| 3 | Optimize & Observe | Full LangGraph agent, model routing, agent decision tracing |
| 4 | Harden & Deploy | Output guardrails, loop detection, confidence-based escalation, AWS ECS Fargate |

## The Agent's Tool Set (Week 3+)

- **Policy & FAQ KB** — RAG retrieval over corpus (this repo's retrieval.py)
- **Customer Record Lookup** — simulated Postgres for order history, account status, tier
- **Order Status Tracker** — simulated API (mock_data/orders.json)
- **Human Escalation** — structured handoff: what was tried, why escalating
- **Response Generator** — structured, cited responses with confidence scoring

## Week 1 Baseline Evaluation Results

### Golden Dataset
- **30 test queries** across 6 intents
- **9 escalation cases** (billing disputes, security issues, damaged deliveries, complex edge cases)
- **21 standard queries** (policy lookups, product info, simple returns)

### 4-Dimensional Evaluation

```bash
# Run full evaluation
python scripts/eval_harness.py

# Run stratified evaluation (classification by intent)
python scripts/eval_harness.py --stratified

# Run both
python scripts/eval_harness.py --both
```

#### Overall Results (Naive Baseline)

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Classification Accuracy** | 90.0% (27/30) | Strong baseline classification |
| **Response Faithfulness** | 4.70/5.0 | Answers well-grounded in retrieved context |
| **Response Correctness** | 3.00/5.0 | Semantic match with expected answers |
| **Routing Accuracy** | 66.7% (20/30) | Never escalates (0% escalation detection) |

#### Per-Intent Classification Accuracy

| Intent | Accuracy | Correct/Total |
|--------|----------|---------------|
| billing_or_payment | 100.0% | 5/5 |
| membership | 100.0% | 5/5 |
| product_info | 100.0% | 4/4 |
| return_or_refund | 83.3% | 5/6 |
| order_status | 80.0% | 4/5 |
| general | 80.0% | 4/5 |

#### Key Findings

**Strengths:**
- High classification accuracy across most intents
- Excellent faithfulness to context (4.70/5.0)
- Perfect accuracy on billing, membership, and product queries

**Areas for Improvement:**
- **0% escalation detection** — naive pipeline never escalates (critical for Week 2+)
- 33% of misclassifications fall back to 'general' intent
- Response correctness could be improved (3.00/5.0)
- Minor confusion between return_or_refund and product_info

**Misclassification Patterns:**
- 1x: return_or_refund → product_info
- 1x: order_status → general  
- 1x: general → membership

### Next Steps (Week 2+)
1. Implement confidence-based escalation logic
2. Add tool use for customer record lookup and order tracking
3. Improve classification for edge cases
4. Build full LangGraph agent with reasoning layer
