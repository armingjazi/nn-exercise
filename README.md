# NN Exercise - LLM-Enhanced User Data Processing

A progressive exercise series demonstrating Python API integration, data processing, LLM capabilities, and agentic workflows using LangChain.

## Overview

This project implements **5 progressive exercises**, each building on concepts from the previous one. The exercises intentionally maintain some code duplication to keep each exercise **self-contained and demonstrable** on its own.

## Progressive Exercise Structure

### Exercise 1: API Integration

**Goal:** Fetch data from Random User API

- Simple HTTP GET request using `requests`
- Basic error handling
- JSON response parsing

**Run:** `python exercise1.py`

---

### Exercise 2: Data Processing & Filtering

**Goal:** Filter and format API data

- Reuses Exercise 1's fetching logic
- Filters users born in 2000 or earlier
- Formats output as array of "FirstName LastName"

**Run:** `python exercise2.py`

---

### Exercise 3: Testing

**Goal:** Ensure code quality with comprehensive tests

- 17 test cases using pytest
- Tests filtering logic, API mocking, edge cases
- Validates Exercise 2 functionality

**Run:** `pytest exercise3.py -v`

---

### Exercise 4: LLM Person Identification P

**Goal:** Identify people using Wikipedia + LLM

**Key Innovation:** Wikipedia-first approach

1. Searches Wikipedia for each person
2. If found � LLM formats the information (verified source)
3. If not found � LLM uses its knowledge (unverified)

**Features:**

-  **Categorized results**: Verified, Unknown, Fictional, Error
- =� **Summary statistics**: Success rate, confidence levels
- =
  **Source attribution**: Shows Wikipedia URLs
- <� **Smart presentation**: Grouped by category with confidence indicators

**Technologies:**

- OpenRouter API (LLM provider)
- Wikipedia REST API (free, unlimited)
- Custom `LLMBackend` and `WikipediaSearch` classes

**Run:** `python exercise4.py`

---

### Exercise 5: LangChain Agentic Workflow

**Goal:** Create an autonomous agent that researches people's best work

**Key Innovation:** Agent with 7 tools + autonomous decision-making

**Agent Tools:**

1. `fetch_users_from_api` - Fetches random users
2. `filter_users_by_birth_year` - Filters by birth year
3. `select_random_people_from_list` - Randomly selects people
4. `search_wikipedia_for_person` - Searches Wikipedia
5. `identify_person_with_llm` - LLM identification (fallback)
6. `check_if_notable` - Assesses notability
7. `research_best_work` - Researches achievements

**Agentic Behavior:**

- Agent autonomously decides which tools to use and when
- Multi-step reasoning: fetch � filter � select � identify � research
- Integrates Wikipedia search for factual accuracy
- Uses LangChain's `create_agent` framework

**Technologies:**

- LangChain (agent framework)
- LangGraph (tool calling)
- `init_chat_model` with OpenRouter

**Run:** `python exercise5.py`

---

## Why Code Duplication Exists

Each exercise is **intentionally standalone** to demonstrate progressive learning:

- **Exercise 1-3**: Foundation (API, processing, testing)
- **Exercise 4**: Adds LLM integration + Wikipedia enhancement
- **Exercise 5**: Refactors into tools for agentic workflow

This approach shows **skill progression** while keeping each exercise independently runnable and demonstrable.

---

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

Get a free API key at [OpenRouter](https://openrouter.ai)

### 3. Run Exercises

```bash
# Run individual exercises
python exercise1.py
python exercise2.py
python exercise4.py  # Requires API key
python exercise5.py  # Requires API key

# Run tests
pytest exercise3.py -v           # 17 tests
pytest test_exercise4.py -v      # Exercise 4 tests
pytest test_exercise5.py -v      # 33 tests for Exercise 5
pytest                           # All tests
```

---

## Technologies Used

| Category            | Technology                      |
| ------------------- | ------------------------------- |
| **Language**        | Python 3.11+                    |
| **API Integration** | requests, Random User API       |
| **LLM**             | OpenRouter (GPT-4o)             |
| **Data Source**     | Wikipedia REST API              |
| **Agent Framework** | LangChain, LangGraph            |
| **Testing**         | pytest, pytest-cov, pytest-mock |
| **Configuration**   | python-dotenv                   |

---

## Performance Notes

**Current (Sequential):**

- 5 people: ~15 seconds
- Each person processed one-by-one

**Potential Optimizations** (documented in ARCHITECTURE.md):

- Async/concurrent processing: 5-10x speedup
- Caching: Instant for repeated queries
- Batch LLM calls: Token efficiency

---

## Documentation

- **ARCHITECTURE.md**: Detailed system architecture with Mermaid diagrams
- **Inline docstrings**: Every function documented
- **Type hints**: Throughout codebase
- **Comments**: Explaining key decisions

---
