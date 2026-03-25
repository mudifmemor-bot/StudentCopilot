# =============================================================
# extensions/nlp_interface.py
# Extension 2: NLP Mini-Interface (Chatbot Style)
#
# CONCEPT:
#   Instead of rigid "enter a number" prompts, the student types
#   naturally. The interface extracts meaning from their words.
#
#   Techniques used:
#     - Keyword spotting  : detect words like "stressed", "deadline"
#     - Pattern matching  : find numbers near keywords ("3 days")
#     - Sentiment scoring : "really stressed" > "a bit stressed"
#     - Slot filling      : track which facts we still need
#     - Targeted questions: only ask what's genuinely missing
#
#   This satisfies: "collects constraints, detects missing info,
#   asks targeted questions" from the coursework brief.
# =============================================================

import re


# -------------------------------------------------------------
# PART 1: KEYWORD AND PATTERN DICTIONARIES
# -------------------------------------------------------------
# These map natural language words to numeric values.
# When a student types "I'm very stressed", we detect "very stressed"
# and map it to a high stress score.

STRESS_KEYWORDS = {
    # High stress indicators
    "overwhelmed":      9,  "panicking":     9,  "panicked":      9,
    "very stressed":    8,  "really stressed":8,  "super stressed": 8,
    "stressed out":     8,  "anxious":       7,  "stressed":       7,
    "worried":          6,  "nervous":       6,  "concerned":      5,
    # Low stress indicators
    "fine":             3,  "okay":          3,  "ok":             3,
    "not stressed":     2,  "calm":          2,  "relaxed":        1,
    "not worried":      2,  "comfortable":   3,
}

CONFIDENCE_KEYWORDS = {
    # Low confidence
    "no idea":          1,  "lost":          1,  "confused":       2,
    "struggling":       2,  "not confident": 2,  "don't understand": 2,
    "behind":           3,  "unsure":        3,
    # Medium
    "okay with it":     5,  "getting there":  5, "some gaps":      4,
    "mostly understand": 6,
    # High confidence
    "confident":        8,  "understand it":  8, "on top of it":   9,
    "fully understand": 9,  "no problem":     8, "got it":         8,
}

URGENCY_KEYWORDS = {
    "today":          1,   "tomorrow":      1,  "tonight":        1,
    "this week":      5,   "few days":      3,  "couple of days": 2,
    "next week":      7,   "soon":          4,  "urgent":         1,
    "not urgent":     10,  "plenty of time": 10,
}

DIFFICULTY_KEYWORDS = {
    "very hard":     3,   "really hard":   3,  "difficult":      3,
    "hard":          3,   "challenging":   3,  "tough":          3,
    "medium":        2,   "moderate":      2,  "okay":           2,
    "manageable":    2,
    "easy":          1,   "simple":        1,  "straightforward":1,
    "not hard":      1,
}


# -------------------------------------------------------------
# PART 2: EXTRACTION FUNCTIONS
# -------------------------------------------------------------
# These scan a message for specific types of information.

def extract_number(text: str,
                   context_words: list = None,
                   default=None):
    """
    Finds a number in the text, optionally near a context word.

    Examples:
      "I have 3 days until deadline" → 3
      "about 10 hours free"          → 10
      "missed 2 sessions"            → 2
    """
    text_lower = text.lower()

    # If context words given, look for number near those words
    if context_words:
        for word in context_words:
            pattern = rf"(\d+(?:\.\d+)?)\s*{re.escape(word)}"
            match   = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))
            # Also try word then number
            pattern2 = rf"{re.escape(word)}\s*(\d+(?:\.\d+)?)"
            match2   = re.search(pattern2, text_lower)
            if match2:
                return float(match2.group(1))

    # Generic: find any number in the text
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text_lower)
    if numbers:
        return float(numbers[0])

    return default


def extract_keyword_value(text: str, keyword_dict: dict, default=None):
    """
    Scans text for keywords from a dictionary and returns
    the associated value. Longer phrases checked first (more specific).

    Example:
      text="I'm very stressed", keyword_dict=STRESS_KEYWORDS → 8
    """
    text_lower = text.lower()
    # Sort by length descending — check longer phrases first
    # so "very stressed" matches before just "stressed"
    for phrase in sorted(keyword_dict.keys(), key=len, reverse=True):
        if phrase in text_lower:
            return keyword_dict[phrase]
    return default


def extract_gender(text: str):
    """Detects gender from natural language."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["female", "woman", "girl", "she", "her"]):
        return "female"
    if any(w in text_lower for w in ["male", "man", "boy", "he", "him"]):
        return "male"
    if any(w in text_lower for w in ["other", "non-binary", "nonbinary", "they"]):
        return "other"
    return None


# -------------------------------------------------------------
# PART 3: MESSAGE PARSER
# -------------------------------------------------------------
# The heart of the NLP interface. Takes one raw message and
# tries to extract as many facts as possible from it.

def parse_message(text: str, context: str = None) -> dict:
    """
    Parses one user message and extracts whatever facts it can.

    Parameters:
        text    — the raw message the student typed
        context — optional hint about what we asked ("stress", "deadline" etc.)

    Returns:
        dict of extracted facts (only keys with values found)
    """
    extracted = {}
    text_lower = text.lower()

    # --- Stress ---
    stress_val = extract_keyword_value(text, STRESS_KEYWORDS)
    if stress_val is None:
        # Try to find a direct number if context was stress
        if context == "stress":
            stress_val = extract_number(text)
    if stress_val is not None:
        extracted["stress_level"] = int(min(10, max(1, stress_val)))

    # --- Confidence ---
    conf_val = extract_keyword_value(text, CONFIDENCE_KEYWORDS)
    if conf_val is None and context == "confidence":
        conf_val = extract_number(text)
    if conf_val is not None:
        extracted["confidence"] = int(min(10, max(1, conf_val)))

    # --- Days until deadline ---
    days_val = extract_number(
        text,
        context_words=["day", "days", "deadline"]
    )
    if days_val is None:
        days_val = extract_keyword_value(text, URGENCY_KEYWORDS)
    if days_val is not None:
        extracted["days_until_deadline"] = int(days_val)

    # --- Hours available ---
    hours_val = extract_number(
        text,
        context_words=["hour", "hours", "free", "available", "study"]
    )
    if hours_val is not None:
        extracted["hours_available"] = int(hours_val)

    # --- Missed sessions ---
    missed_val = extract_number(
        text,
        context_words=["missed", "skipped", "session", "sessions", "class"]
    )
    if missed_val is not None:
        extracted["missed_sessions"] = int(missed_val)

    # --- Gender ---
    gender_val = extract_gender(text)
    if gender_val:
        extracted["gender"] = gender_val

    # --- Name detection ---
    name_match = re.search(
        r"(?:my name is|i(?:'m| am) called|call me)\s+([A-Z][a-z]+)",
        text,
        re.IGNORECASE
    )
    if name_match:
        extracted["name"] = name_match.group(1)

    return extracted


# -------------------------------------------------------------
# PART 4: SLOT TRACKER
# -------------------------------------------------------------
# Tracks which facts we still need and generates targeted
# questions only for missing ones.

REQUIRED_FACTS = [
    "name", "gender", "stress_level", "confidence",
    "days_until_deadline", "hours_available", "missed_sessions"
]

TARGETED_QUESTIONS = {
    "name":                "What's your name?",
    "gender":              "What's your gender? (female / male / other)",
    "stress_level":        "How stressed are you feeling right now? (1=calm, 10=overwhelmed)",
    "confidence":          "How confident are you with your current material? (1=lost, 10=fully on top)",
    "days_until_deadline": "How many days until your next major deadline?",
    "hours_available":     "How many hours can you study this week?",
    "missed_sessions":     "How many study sessions have you missed recently?",
}

def get_next_question(facts: dict) -> tuple:
    """
    Returns the next (fact_key, question) to ask based on
    what's still missing from the facts dictionary.
    Returns (None, None) if all required facts are collected.
    """
    for fact in REQUIRED_FACTS:
        if fact not in facts:
            return fact, TARGETED_QUESTIONS[fact]
    return None, None


# -------------------------------------------------------------
# PART 5: CHATBOT RESPONSE GENERATOR
# -------------------------------------------------------------
# Produces friendly, context-aware responses — not robotic prompts.

def generate_response(facts: dict,
                      last_extracted: dict,
                      next_fact: str) -> str:
    """
    Generates a natural-sounding chatbot response.

    Acknowledges what was just extracted, then asks the next question.
    """
    acknowledgements = []

    # Acknowledge what we just learned
    if "stress_level" in last_extracted:
        s = last_extracted["stress_level"]
        if s >= 7:
            acknowledgements.append(
                f"I can hear that you're under a lot of pressure (stress {s}/10)."
            )
        elif s >= 4:
            acknowledgements.append(f"Got it — moderate stress level ({s}/10).")
        else:
            acknowledgements.append(
                f"Good, sounds like you're relatively calm ({s}/10)."
            )

    if "days_until_deadline" in last_extracted:
        d = last_extracted["days_until_deadline"]
        if d <= 2:
            acknowledgements.append(
                f"That deadline is very close — {d} day(s) away. Let's move quickly."
            )
        elif d <= 5:
            acknowledgements.append(f"Deadline in {d} days — we have a bit of time to plan.")
        else:
            acknowledgements.append(f"Deadline in {d} days — reasonable amount of time.")

    if "confidence" in last_extracted:
        c = last_extracted["confidence"]
        if c <= 3:
            acknowledgements.append(
                f"Confidence at {c}/10 — we'll make sure the plan addresses this."
            )

    if "name" in last_extracted:
        acknowledgements.append(
            f"Nice to meet you, {last_extracted['name']}!"
        )

    # Build the response
    parts = acknowledgements if acknowledgements else []

    if next_fact:
        question = TARGETED_QUESTIONS[next_fact]
        parts.append(question)
        return " ".join(parts)
    else:
        return " ".join(parts) + " Great — I have everything I need. Building your plan now..."


# -------------------------------------------------------------
# PART 6: MAIN CHATBOT LOOP
# -------------------------------------------------------------

def run_chatbot() -> dict:
    """
    Runs the full NLP chatbot interface.
    Collects all required student facts through natural conversation.

    Returns:
        dict of collected student facts
    """
    print("\n" + "=" * 55)
    print("  STUDENT COPILOT — NLP Chat Interface")
    print("=" * 55)
    print("\n  Hi! I'm your study copilot. Tell me about your situation")
    print("  and I'll build a personalised plan for you.")
    print("  (Type naturally — no need for exact numbers unless you want)\n")

    facts   = {}
    history = []

    # Opening prompt
    print("  Copilot: What's going on with your studies right now?")

    while True:
        # Get next required fact
        next_fact, _ = get_next_question(facts)

        if next_fact is None:
            # All facts collected
            print("\n  Copilot: Perfect — I have everything I need!")
            print("  Copilot: Building your personalised study plan...\n")
            break

        # Get student input
        user_input = input("  You: ").strip()
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        # Parse the message
        extracted = parse_message(user_input, context=next_fact)

        # Update facts with extracted values
        facts.update(extracted)

        # Get what's still missing
        next_fact_after, _ = get_next_question(facts)

        # Generate and print response
        response = generate_response(facts, extracted, next_fact_after)
        print(f"\n  Copilot: {response}\n")
        history.append({"role": "copilot", "content": response})

        # If nothing was extracted, ask the question more directly
        if not extracted:
            _, direct_q = get_next_question(facts)
            if direct_q:
                print(f"  Copilot: Let me ask more directly — {direct_q}\n")

    # Fill any remaining gaps with defaults (safety net)
    defaults = {
        "name":                "Student",
        "gender":              "other",
        "stress_level":        5,
        "confidence":          5,
        "days_until_deadline": 7,
        "hours_available":     10,
        "missed_sessions":     0,
    }
    for key, val in defaults.items():
        if key not in facts:
            facts[key] = val

    print("  Collected facts:")
    for k, v in facts.items():
        print(f"    {k}: {v}")

    return facts


# -------------------------------------------------------------
# TEST — demonstrate NLP extraction without interactive input
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  NLP EXTRACTION DEMO")
    print("=" * 55)

    test_messages = [
        "I'm really stressed, my deadline is in 2 days",
        "I have about 8 hours free this week and I'm pretty confident",
        "My name is Sara, I'm female, I've missed 3 sessions and I feel overwhelmed",
        "Not too worried, exam is next week, I understand the material",
        "Deadline tomorrow and I have no idea what I'm doing, only 3 hours free",
    ]

    for msg in test_messages:
        print(f"\n  Input : \"{msg}\"")
        result = parse_message(msg)
        print(f"  Parsed: {result}")

    print("\n  (Run run_chatbot() for the interactive version)")
