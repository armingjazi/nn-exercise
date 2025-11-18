# Architecture Flow Documentation

This document provides comprehensive architecture diagrams for the NN Exercise project, showing the flow of data and interactions between components across all exercises.

## Table of Contents

1. [Exercise 1: Data Fetching](#exercise-1-data-fetching)
2. [Exercise 4: LLM Person Identification](#exercise-4-llm-person-identification)
3. [Exercise 5: LangChain Agentic Workflow](#exercise-5-langchain-agentic-workflow)

## Exercise 1: Data Fetching

Simple API integration to fetch random user data.

```mermaid
graph TD
    A[Start Exercise 1] --> B[fetch_random_users function]
    B --> C{Make API Request}
    C -->|GET| D[https://randomuser.me/api/?results=20]
    D -->|Success| E[Parse JSON Response]
    D -->|Error| F[Raise RequestException]
    E --> G[Return dict with 20 users]
    G --> H[Display Sample User]
    H --> I[End]

    style D fill:#e1f5ff,stroke:#333,stroke-width:2px
    style G fill:#d4edda,stroke:#333,stroke-width:2px
    style F fill:#f8d7da,stroke:#333,stroke-width:2px
```

**Key Components:**

- `fetch_random_users()`: Makes HTTP GET request
- Error handling for network issues
- Returns raw JSON response

---

## Exercise 4: LLM Person Identification

Multi-step identification using Wikipedia + LLM.

```mermaid
sequenceDiagram
    participant Main as exercise4.py
    participant API as Random User API
    participant Filter as format_and_filter_users
    participant LLM as LLMBackend
    participant Wiki as Wikipedia API
    participant OR as OpenRouter
    participant Display as display_identifications

    Main->>API: GET /api/?results=20
    API-->>Main: 20 random users

    Main->>Filter: filter by birth_year <= 2000
    Filter-->>Main: Filtered user list

    Main->>Main: select_random_people(5)

    Note over Main,OR: For each of 5 people

    loop For each person
        Main->>LLM: identify_person_with_search(first, last)

        LLM->>Wiki: search_person(first, last)
        Wiki-->>LLM: Wikipedia search result

        alt Wikipedia article found
            LLM->>OR: Summarize Wikipedia content
            OR-->>LLM: Formatted summary
            LLM-->>Main: ✅ Verified info + source URL
        else No Wikipedia article
            LLM->>OR: Identify using LLM knowledge
            OR-->>LLM: LLM response
            LLM-->>Main: ❌ Unverified info + note
        end
    end

    Main->>Display: display_identifications(results)
    Display->>Display: categorize_identification()
    Display->>Display: display_summary_statistics()
    Display-->>Main: Formatted categorized output
```

**Key Features:**

- **Wikipedia-first approach**: Searches Wikipedia before using LLM
- **Categorization**: Verified, Unknown, Fictional, Error
- **Source attribution**: Shows Wikipedia URLs when available
- **Fallback mechanism**: Uses LLM knowledge if Wikipedia fails

---

## Exercise 5: LangChain Agentic Workflow

Autonomous agent with tool-calling capabilities.

```mermaid
graph TB
    A[User Request] --> B[BestWorkAgent]
    B --> C{LangChain Agent<br/>with init_chat_model}

    C --> D[Available Tools]

    D --> T1[fetch_users_from_api]
    D --> T2[filter_users_by_birth_year]
    D --> T3[select_random_people_from_list]
    D --> T4[search_wikipedia_for_person]
    D --> T5[identify_person_with_llm]
    D --> T6[check_if_notable]
    D --> T7[research_best_work]

    T1 --> E[Random User API]
    T2 --> F[Filtering Logic]
    T3 --> G[Random Selection]
    T4 --> H[Wikipedia API]
    T5 --> I[OpenRouter LLM]
    T6 --> J[Notability Assessment]
    T7 --> K[LLM Research]

    E --> L{Agent Decision Loop}
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L

    L -->|Tool Selection| C
    L -->|Task Complete| M[Final Report]

    style C fill:#ffc107,stroke:#333,stroke-width:3px
    style L fill:#17a2b8,stroke:#333,stroke-width:2px
    style M fill:#28a745,stroke:#333,stroke-width:2px
```

**Agentic Capabilities:**

1. **Autonomous tool selection**: Agent decides which tools to use
2. **Multi-step reasoning**: Chains multiple tool calls together
3. **Iterative refinement**: Can retry with different approaches
4. **Context awareness**: Maintains conversation state

---

## Performance Considerations

Current bottlenecks and optimization opportunities:

```mermaid
graph LR
    A[Sequential Processing] -->|Bottleneck| B[5-15 seconds<br/>for 5 people]

    C[Async Processing] -->|Optimized| D[2-3 seconds<br/>for 5 people]

    E[No Caching] -->|Bottleneck| F[Repeated API calls<br/>for same person]

    G[File-based Cache] -->|Optimized| H[Instant response<br/>for cached results]

    style A fill:#f8d7da,stroke:#333,stroke-width:2px
    style C fill:#d4edda,stroke:#333,stroke-width:2px
    style E fill:#f8d7da,stroke:#333,stroke-width:2px
    style G fill:#d4edda,stroke:#333,stroke-width:2px
```

**Optimization Strategies:**

1. **Async/Concurrent Processing**: Use `asyncio` for parallel API calls (5-10x speedup)
2. **Caching**: Cache Wikipedia + LLM results (instant for repeated queries)
3. **Connection Pooling**: Reuse HTTP connections (small latency reduction)
4. **Batch Optimization**: Process multiple people in single LLM call (token efficiency)

---

## Error Handling Flow

```mermaid
stateDiagram-v2
    [*] --> FetchData
    FetchData --> ParseData: Success
    FetchData --> APIError: Network Error

    ParseData --> FilterData: Valid JSON
    ParseData --> ParseError: Invalid JSON

    FilterData --> SearchWikipedia: Data Ready

    SearchWikipedia --> WikipediaFound: Article Exists
    SearchWikipedia --> WikipediaNotFound: No Article
    SearchWikipedia --> WikipediaError: API Error

    WikipediaFound --> LLMSummarize
    WikipediaNotFound --> LLMIdentify

    LLMSummarize --> Success: ✅ Verified
    LLMIdentify --> Success: ❌ Unverified
    WikipediaError --> Success: ⚠️ Error + Fallback

    APIError --> [*]: Display Error
    ParseError --> [*]: Display Error
    Success --> [*]: Display Results
```

---

## Security Considerations

```mermaid
graph TD
    A[API Key Management] --> B[.env File]
    B --> C[.gitignore Protection]
    C --> D[Config Module Validation]

    E[Input Validation] --> F[Name Validation]
    F --> G[Regex Patterns]

    H[Rate Limiting] --> I[Respect API Limits]
    I --> J[5 people max for LLM]

    K[Error Handling] --> L[No Sensitive Data in Logs]
    L --> M[Generic Error Messages]

    style B fill:#f8d7da,stroke:#333,stroke-width:2px
    style C fill:#d4edda,stroke:#333,stroke-width:2px
    style D fill:#d4edda,stroke:#333,stroke-width:2px
```
