        # Day 08 Lab Report

        ## 1. Team / student

        - Name:
        - Repo/commit:
        - Date:

        ## 2. Architecture

        This lab uses a LangGraph `StateGraph` for a support-ticket workflow. The graph starts
        at intake, classifies the query with an LLM, then routes to one of five paths: simple
        answer, tool lookup, clarification, risky action approval, or retry/error handling.
        All paths terminate through `finalize` before `END` for consistent audit logging.

        ## 3. State schema

        Important state fields are split between overwrite fields and append-only audit fields.

        | Field | Reducer | Why |
        |---|---|---|
        | query | overwrite | normalized current user request |
        | route | overwrite | current classified route |
        | risk_level | overwrite | current risk status |
        | attempt | overwrite | retry loop counter |
        | max_attempts | overwrite | retry safety bound |
        | final_answer | overwrite | final response to user |
        | evaluation_result | overwrite | gate for retry vs answer |
        | pending_question | overwrite | clarification prompt |
        | proposed_action | overwrite | risky action summary |
        | approval | overwrite | approval decision payload |
        | messages | append | lightweight trace of conversation flow |
        | tool_results | append | record of tool outputs across retries |
        | errors | append | retry and failure history |
        | events | append | structured audit trail for grading/debugging |

        ## 4. Scenario results

        ### Summary

        | Metric | Value |
        |---|---:|
        | Total scenarios | 7 |
        | Success rate | 100.00% |
        | Average nodes visited | 6.43 |
        | Total retries | 3 |
        | Total interrupts | 2 |

        ### Per-scenario results

        | Scenario | Expected route | Actual route | Success | Retries | Interrupts |
        |---|---|---|---:|---:|---:|
        | S01_simple | simple | simple | yes | 0 | 0 |
| S02_tool | tool | tool | yes | 0 | 0 |
| S03_missing | missing_info | missing_info | yes | 0 | 0 |
| S04_risky | risky | risky | yes | 0 | 1 |
| S05_error | error | error | yes | 2 | 0 |
| S06_delete | risky | risky | yes | 0 | 1 |
| S07_dead_letter | error | error | yes | 1 | 0 |

        ## 5. Failure analysis

        1. Retry or tool failure:
           The graph can enter a bounded retry loop when tool evaluation returns `needs_retry`.
           The `attempt < max_attempts` check prevents infinite loops and sends exhausted cases
           to `dead_letter`.

        2. Risky action without approval:
           Risky requests are routed through `risky_action` and `approval` before tool execution.
           If approval is rejected, the graph routes to clarification instead of executing the action.

        ## 6. Persistence / recovery evidence

        The workflow is invoked with a `thread_id` per scenario run. The configured checkpointer
        can persist state to SQLite, which provides a durable base for local recovery and state
        inspection across runs.

        ## 7. Extension work

        Completed extension work includes SQLite-backed checkpoint persistence for local state
        durability in addition to the core orchestration path, metrics output, and report generation.

        ## 8. Improvement plan

        If given one more day, the first production improvements would be:
        - add state history inspection and recovery walkthrough screenshots
        - improve `evaluate_node` with LLM-as-judge scoring
        - strengthen approval flow with real human interrupt handling
        - expand scenario coverage with more edge cases and hidden-failure style tests
