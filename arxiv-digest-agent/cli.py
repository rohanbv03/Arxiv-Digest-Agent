#!/usr/bin/env python3
"""
Usage:
    python cli.py "recent work on KV-cache compression for LLMs"
    python cli.py 2401.12345
    python cli.py https://arxiv.org/abs/2401.12345

    # non-interactive demo mode (skips input(), asks fixed questions, then exits):
    python cli.py 2401.12345 --questions "What is the main contribution?" "What eviction policy do they use?"
"""
import argparse
import sys
import uuid

from arxiv_agent.graph import build_pipeline_graph, build_qa_graph
from arxiv_agent.qa import format_sources


def interactive_select(scored_candidates):
    print(f"\nFound {len(scored_candidates)} candidate papers:\n")
    for i, sc in enumerate(scored_candidates):
        c = sc["paper"]
        print(f"  [{i}] ({sc['score']:.2f}) {c['title']}  ({c['arxiv_id']}, {c['published']})")
    choice = input(f"\nPick a paper to summarize [0-{len(scored_candidates)-1}, default 0]: ").strip()
    return int(choice) if choice.isdigit() else 0


def run_qa_turn(qa_graph, pipeline_state: dict, question: str) -> dict:
    qa_state = {
        "pending_question": question,
        "collection_name": pipeline_state["collection_name"],
    }
    result = qa_graph.invoke(qa_state)
    print(f"\n{result['qa_answer']}")
    sources = format_sources(result.get("qa_sources", []))
    if sources:
        print(sources)
    print()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="topic, arXiv ID, or arXiv URL")
    parser.add_argument(
        "--questions", nargs="*", default=None,
        help="run these questions non-interactively instead of dropping into the input() loop",
    )
    args = parser.parse_args()

    pipeline = build_pipeline_graph(on_select=interactive_select if args.questions is None else (lambda c: 0))
    qa_graph = build_qa_graph()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 200}
    final_state = pipeline.invoke({"user_input": args.query}, config=config)

    if final_state.get("error"):
        kind = final_state.get("error_kind", "error")
        print(f"\n[{kind}] {final_state['error']}")
        sys.exit(1)

    if args.questions is not None:
        for q in args.questions:
            print(f"\nQ: {q}")
            run_qa_turn(qa_graph, final_state, q)
        return

    while True:
        question = input("Ask a question about this paper (or type 'exit' to quit): ").strip()
        if question.lower() in {"exit", "quit", "q", ""}:
            break
        run_qa_turn(qa_graph, final_state, question)


if __name__ == "__main__":
    main()
