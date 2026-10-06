"""Ask the assistant from the terminal.

    uv run python ask.py "How much is a deep clean for a 3-bedroom home?"
    uv run python ask.py --sources "Can I cancel the day before?"
"""

from __future__ import annotations

import argparse

import anthropic

from rag.answer import MissingAPIKeyError
from rag.pipeline import Assistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask BrightClean's document assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--sources", action="store_true", help="also print the retrieved chunks")
    args = parser.parse_args()

    try:
        answer, hits = Assistant().ask(args.question)
    except MissingAPIKeyError as e:
        raise SystemExit(f"Error: {e}")
    except anthropic.AuthenticationError:
        raise SystemExit("Error: the API key in .env is invalid. Set a valid ANTHROPIC_API_KEY and try again.")
    except anthropic.APIConnectionError:
        raise SystemExit("Error: couldn't reach the Claude API. Check your internet connection and try again.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"Error: the Claude API returned an error ({e.status_code}). Please try again.")
    print(answer.text)
    for c in answer.citations:
        print(f'\n[{c.number}] {c.label}\n    "{c.quote}"')
    if args.sources:
        print("\nRetrieved chunks:")
        for h in hits:
            print(f"  {h.similarity:.2f}  {h.chunk.label}")


if __name__ == "__main__":
    main()
