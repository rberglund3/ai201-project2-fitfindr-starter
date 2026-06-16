"""
agent.py

The FitFindr planning loop. Orchestrates the three tools in response to a
natural language user query, passing state between them via a session dict.

Complete tools.py and test each tool in isolation before implementing this file.

Usage (once implemented):
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import json
import re

import tools
from tools import search_listings, suggest_outfit, create_fit_card


# ── query parsing ───────────────────────────────────────────────────────────────

_PARSE_SYSTEM = (
    "You extract structured search parameters from a shopper's request for "
    "secondhand clothing. Respond with ONLY a JSON object — no prose, no code "
    "fences — with exactly these keys:\n"
    '  "description": string of keywords describing the item (never null),\n'
    '  "size": the size string if one is mentioned, else null,\n'
    '  "max_price": a number if a price ceiling is mentioned, else null.\n'
    'Example: {"description": "vintage graphic tee", "size": "M", "max_price": 30}'
)


def _parse_query_regex(query: str) -> dict:
    """Deterministic fallback parser used when the LLM is unavailable."""
    # Max price: "$30", "under 30", "30 dollars", etc.
    price = None
    price_match = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars|usd|bucks)?", query)
    if price_match and re.search(r"under|less than|below|cheaper|\$|max|budget", query, re.I):
        price = float(price_match.group(1))

    # Size: "size M", "size US 9", "in a medium".
    size = None
    size_match = re.search(r"\bsize\s+([A-Za-z0-9/ ]+?)(?:\s*(?:under|below|less|$|,|\.))", query, re.I)
    if size_match:
        size = size_match.group(1).strip()

    return {"description": query.strip(), "size": size, "max_price": price}


def _parse_query(query: str) -> dict:
    """
    Extract {description, size, max_price} from a natural language query.

    Tries the Groq LLM for robust parsing and falls back to a regex parser if
    the call fails or returns something unusable. Always returns a dict with all
    three keys, with a non-empty description.
    """
    try:
        raw = tools._chat(
            [
                {"role": "system", "content": _PARSE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        # Strip any accidental code fences before parsing.
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw)

        description = (parsed.get("description") or "").strip() or query.strip()
        size = parsed.get("size")
        size = str(size).strip() if size else None
        max_price = parsed.get("max_price")
        max_price = float(max_price) if isinstance(max_price, (int, float)) else None

        return {"description": description, "size": size, "max_price": max_price}
    except Exception:
        return _parse_query_regex(query)


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.

    You may add fields to this dict as needed for your implementation.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "selected_item": None,       # top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "error": None,               # set if the interaction ended early
    }


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — use get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and the other
        output fields (outfit_suggestion, fit_card) will be None.

    TODO — implement this function using the planning loop you designed in planning.md:

        Step 1: Initialize the session with _new_session().

        Step 2: Parse the user's query to extract a description, size, and
                max_price. You can use regex, string splitting, or ask the LLM
                to parse it — document your choice in planning.md.
                Store the result in session["parsed"].

        Step 3: Call search_listings() with the parsed parameters.
                Store results in session["search_results"].
                If no results: set session["error"] to a helpful message and
                return the session early. Do NOT proceed to suggest_outfit
                with empty input.

        Step 4: Select the item to use (e.g., the top result).
                Store it in session["selected_item"].

        Step 5: Call suggest_outfit() with the selected item and wardrobe.
                Store the result in session["outfit_suggestion"].

        Step 6: Call create_fit_card() with the outfit suggestion and selected item.
                Store the result in session["fit_card"].

        Step 7: Return the session.

    Before writing code, complete the Planning Loop and State Management sections
    of planning.md — your implementation should match what you described there.
    """
    # Step 1: fresh session — the single source of truth for this interaction.
    session = _new_session(query, wardrobe)

    # Step 2: parse the natural language query into search parameters.
    parsed = _parse_query(query)
    session["parsed"] = parsed

    # Step 3: search the listings with the parsed parameters.
    results = search_listings(
        description=parsed["description"],
        size=parsed["size"],
        max_price=parsed["max_price"],
    )
    session["search_results"] = results

    # Conditional branch: no matches → friendly error, return early.
    # No other tools are called in this case.
    if not results:
        session["error"] = (
            "Sorry, I couldn't find any listings matching that. Try loosening "
            "the price, size, or describing the item a little differently."
        )
        return session

    # Step 4: select the top result and thread it through the next tools.
    session["selected_item"] = results[0]

    # Step 5: suggest an outfit using the selected item and the user's wardrobe.
    session["outfit_suggestion"] = suggest_outfit(
        session["selected_item"], wardrobe
    )

    # Step 6: build a shareable fit card from the outfit and the selected item.
    session["fit_card"] = create_fit_card(
        session["outfit_suggestion"], session["selected_item"]
    )

    # Step 7: return the populated session.
    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        print(f"Found: {session['selected_item']['title']}")
        print(f"\nOutfit: {session['outfit_suggestion']}")
        print(f"\nFit card: {session['fit_card']}")

    print("\n\n=== No-results path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error message: {session2['error']}")
