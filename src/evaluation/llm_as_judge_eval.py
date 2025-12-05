# eval_cragmm.py
# pip install openai>=1.14.0 tiktoken

from __future__ import annotations
from typing import Tuple, Literal, Dict, Any
import json, os, openai, time, re

# --------------------------------------------------------------------------
# Configuration – edit if necessary
# --------------------------------------------------------------------------
MODEL_NAME       = "gpt-4o-mini"          # or "gpt-4o", "gpt-4-turbo", …
TEMPERATURE      = 0.0                    # deterministic rubric application
TIMEOUT_SECONDS  = 60
MAX_RETRIES      = 3                      # network / rate-limit resilience

# Make sure OPENAI_API_KEY is set in your environment.
openai_client = openai.Client(timeout=TIMEOUT_SECONDS)


# --------------------------------------------------------------------------
# Prompt template the grader sees
# --------------------------------------------------------------------------
SYSTEM_MSG = """
You are an impartial grader. Compare a model's answer with the reference
answer and decide which of the following four categories best describes
the model answer:

• Perfect      → Score  1.0
• Acceptable   → Score  0.5
• Missing      → Score  0.0
• Incorrect    → Score -1.0

Rules:
1. A Perfect answer fully satisfies the user question and adds **no
   hallucinated content** beyond what can be inferred from the reference.
2. An Acceptable answer is mostly correct / helpful but has small errors
   that do not invalidate the usefulness of the response.
3. Missing means the model effectively refused or gave no useful info.
4. Incorrect means the response contains wrong or irrelevant info that
   could mislead the user.

Return your result as **one JSON object** with keys:
  "score"       : one of 1.0, 0.5, 0.0, -1.0
  "explanation" : short (≤ 50 words) reason for the score.
Do NOT output anything else.
""".strip()


# --------------------------------------------------------------------------
# Main public helper
# --------------------------------------------------------------------------

def evaluate_answer(
    predicted_text: str,
    ground_truth_text: str,
    model: str = MODEL_NAME,
    **chat_completion_kwargs: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Compare `predicted_text` with `ground_truth_text` via GPT-4o
    and return (score, explanation).

    Raises a ValueError if the model’s JSON output cannot be parsed.
    Additional kwargs are forwarded to openai.chat.completions.create().
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": (
                f"REFERENCE_ANSWER:\n{ground_truth_text}\n\n"
                f"MODEL_ANSWER:\n{predicted_text}\n\n"
                "Grade the MODEL_ANSWER now."
            ),
        },
    ]

    resp = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        **chat_completion_kwargs,
    )

    # --------------- Parse the assistant JSON ----------------------------
    # try:
    #     raw = resp.choices[0].message.content.strip()
    #     result = json.loads(raw)
    #     score = float(result["score"])
    #     explanation = str(result["explanation"]).strip()
    # except (KeyError, json.JSONDecodeError, ValueError) as exc:
    #     raise ValueError(
    #         f"Could not parse grading JSON from model:\n{resp.choices[0].message.content}"
    #     ) from exc

    # if score not in (1.0, 0.5, 0.0, -1.0):
    #     raise ValueError(f"Illegal score {score} returned by the grader LLM.")

    raw = resp.choices[0].message.content
    try:
        parsed = _coerce_json(raw)
    except ValueError as e:
        # Log the raw response for debugging, then propagate
        print("⚠️  Failed to parse grader output:\n", raw)
        raise

    score = float(parsed["score"])
    explanation = str(parsed["explanation"]).strip()

    if score not in (1.0, 0.5, 0.0, -1.0):
        raise ValueError(f"Illegal score {score} parsed from grader.")

    return score, explanation


def _coerce_json(text: str) -> Dict[str, Any]:
    """
    Try vanilla json.loads first. If it fails, attempt to
    strip code fences or grab the first {...} block.
    """
    text = text.strip()

    # 1. direct parse --------------------------------------------------
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. strip markdown ``` fences ------------------------------------
    if text.startswith("```"):
        # remove ```lang   ...   ``` (first and last line with backticks)
        text = re.sub(r"^```.*?\n", "", text, count=1, flags=re.DOTALL)
        text = re.sub(r"\n```$", "", text, count=1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

    # 3. extract first {...} ------------------------------------------
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not coerce grader output to JSON.")



# --------------------------------------------------------------------------
# Quick smoke-test (remove / adapt for your unit tests)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    gtruth = "Paris is the capital city of France."
    good    = "The capital of France is Paris."
    mediocre = "It’s probably Paris, I think."
    missing  = "I’m not sure about that, sorry!"
    wrong    = "The capital of France is Marseille."

    for candidate in (good, mediocre, missing, wrong):
        sc, exp = evaluate_answer(candidate, gtruth)
        print(f"{candidate[:35]:35} -> {sc:+4.1f}  | {exp}")
