# =============================================================
# rules/expert.py
# Student Success Copilot — Rule-Based Expert System
#
# WHAT THIS FILE DOES:
#   1. Defines a set of IF-THEN rules with confidence scores
#   2. Forward chaining  — starts from facts, fires rules,
#                          derives new facts until no more rules fire
#   3. Backward chaining — starts from a goal, works backward to
#                          find what information is missing, then
#                          asks the user a targeted question
#   4. Returns a full reasoning trace so the system can explain
#      every decision it made, step by step
#
# KEY CONCEPT — WHY TWO TYPES OF CHAINING?
#   Forward : "Here are the facts — what can I conclude?"
#   Backward: "I want to conclude X — what facts do I still need?"
#   Together they make the system both proactive (forward) and
#   smart about what it asks (backward).
# =============================================================


# -------------------------------------------------------------
# PART 1: THE KNOWLEDGE BASE — all the rules
# -------------------------------------------------------------
# Each rule is a dictionary with:
#   name        — short label for the trace log
#   conditions  — list of (fact_key, operator, value) tuples
#   conclusion  — the new fact this rule adds to working memory
#   confidence  — how strongly this rule supports its conclusion (0.0–1.0)
#   explanation — plain English sentence for the output
#
# OPERATOR KEY:
#   "lte" = less than or equal    (<=)
#   "gte" = greater than or equal (>=)
#   "lt"  = less than             (<)
#   "gt"  = greater than          (>)
#   "eq"  = equal                 (==)
#   "in"  = value is in a list

RULES = [
    # --- URGENCY RULES ---
    # These detect time pressure situations

    {
        "name": "Rule-1: Critical deadline",
        "conditions": [("days_until_deadline", "lte", 3)],
        "conclusion": "deadline_critical",
        "confidence": 0.9,
        "explanation": "Deadline is critically close (within 3 days)"
    },
    {
        "name": "Rule-2: Approaching deadline",
        "conditions": [("days_until_deadline", "lte", 5),
                       ("days_until_deadline", "gt", 3)],
        "conclusion": "deadline_approaching",
        "confidence": 0.6,
        "explanation": "Deadline is approaching (within 5 days)"
    },

    # --- STRESS RULES ---

    {
        "name": "Rule-3: High stress",
        "conditions": [("stress_level", "gte", 7)],
        "conclusion": "high_stress",
        "confidence": 0.8,
        "explanation": "Student is experiencing high stress"
    },
    {
        "name": "Rule-4: Moderate stress",
        "conditions": [("stress_level", "gte", 4),
                       ("stress_level", "lt", 7)],
        "conclusion": "moderate_stress",
        "confidence": 0.5,
        "explanation": "Student is experiencing moderate stress"
    },

    # --- CONFIDENCE RULES ---

    {
        "name": "Rule-5: Low confidence",
        "conditions": [("confidence", "lte", 4)],
        "conclusion": "low_confidence",
        "confidence": 0.7,
        "explanation": "Student has low confidence in the material"
    },

    # --- ATTENDANCE RULES ---

    {
        "name": "Rule-6: Poor attendance",
        "conditions": [("missed_sessions", "gte", 3)],
        "conclusion": "poor_attendance",
        "confidence": 0.8,
        "explanation": "Student has missed multiple sessions recently"
    },
    {
        "name": "Rule-7: Some missed sessions",
        "conditions": [("missed_sessions", "gte", 2),
                       ("missed_sessions", "lt", 3)],
        "conclusion": "some_missed_sessions",
        "confidence": 0.5,
        "explanation": "Student has missed some sessions"
    },

    # --- TIME AVAILABILITY ---

    {
        "name": "Rule-8: Limited study time",
        "conditions": [("hours_available", "lte", 5)],
        "conclusion": "limited_time",
        "confidence": 0.6,
        "explanation": "Student has very limited study time this week"
    },

    # --- COMPOUND RULES (chain from derived facts above) ---
    # These fire AFTER the simple rules above have derived new facts.
    # This is the "chaining" — conclusions become inputs for new rules.

    {
        "name": "Rule-9: Urgent + high stress → HIGH risk",
        "conditions": [("deadline_critical", "eq", True),
                       ("high_stress",        "eq", True)],
        "conclusion": "risk_HIGH",
        "confidence": 0.9,
        "explanation": "Critical deadline combined with high stress strongly indicates HIGH risk"
    },
    {
        "name": "Rule-10: Urgent + low confidence → HIGH risk",
        "conditions": [("deadline_critical", "eq", True),
                       ("low_confidence",    "eq", True)],
        "conclusion": "risk_HIGH",
        "confidence": 0.85,
        "explanation": "Critical deadline with low confidence indicates HIGH risk"
    },
    {
        "name": "Rule-11: Poor attendance + high stress → HIGH risk",
        "conditions": [("poor_attendance", "eq", True),
                       ("high_stress",     "eq", True)],
        "conclusion": "risk_HIGH",
        "confidence": 0.8,
        "explanation": "Missing sessions while stressed indicates HIGH risk"
    },
    {
        "name": "Rule-12: Approaching deadline + stress → MEDIUM risk",
        "conditions": [("deadline_approaching", "eq", True),
                       ("moderate_stress",      "eq", True)],
        "conclusion": "risk_MEDIUM",
        "confidence": 0.65,
        "explanation": "Upcoming deadline with moderate stress indicates MEDIUM risk"
    },
    {
        "name": "Rule-13: Limited time + approaching deadline → MEDIUM risk",
        "conditions": [("limited_time",         "eq", True),
                       ("deadline_approaching",  "eq", True)],
        "conclusion": "risk_MEDIUM",
        "confidence": 0.6,
        "explanation": "Limited study time before an approaching deadline is a warning sign"
    },
    {
        "name": "Rule-14: Some missed sessions + low confidence → MEDIUM risk",
        "conditions": [("some_missed_sessions", "eq", True),
                       ("low_confidence",       "eq", True)],
        "conclusion": "risk_MEDIUM",
        "confidence": 0.6,
        "explanation": "Missed sessions combined with low confidence suggests MEDIUM risk"
    },
]


# -------------------------------------------------------------
# PART 2: THE CONDITION EVALUATOR
# -------------------------------------------------------------
# This helper checks a single condition against the current facts.
# "working_memory" is just a dictionary of everything we know so far.

def _check_condition(condition: tuple, working_memory: dict) -> bool:
    """
    Evaluates one condition tuple against working memory.
    Example: ("stress_level", "gte", 7) with {"stress_level": 9} → True
    """
    fact_key, operator, value = condition

    # If we don't know this fact yet, the condition can't be checked
    if fact_key not in working_memory:
        return False

    actual = working_memory[fact_key]

    if operator == "lte": return actual <= value
    if operator == "gte": return actual >= value
    if operator == "lt":  return actual < value
    if operator == "gt":  return actual > value
    if operator == "eq":  return actual == value
    if operator == "in":  return actual in value
    return False


# -------------------------------------------------------------
# PART 3: FORWARD CHAINING
# -------------------------------------------------------------
# Algorithm:
#   1. Start with the student's raw facts in working memory
#   2. Go through ALL rules
#   3. If a rule's conditions are ALL met → fire it (add conclusion to memory)
#   4. Keep repeating until no new facts are added (fixed point)
#   5. Return everything we derived + the full reasoning trace

def forward_chain(student_facts: dict) -> dict:
    """
    Runs forward chaining on a student's facts.

    Parameters:
        student_facts — dict of known facts about the student

    Returns:
        dict with keys:
            working_memory  — all facts after chaining (original + derived)
            fired_rules     — list of rule names that fired
            trace           — step-by-step reasoning log (for explanation)
            risk_level      — final risk conclusion ("High"/"Medium"/"Low")
            confidence      — combined confidence score
            explanations    — list of plain-English reason strings
    """
    # Working memory starts with what we know about the student
    working_memory = dict(student_facts)

    fired_rules  = []
    trace        = []
    explanations = []
    risk_scores  = {"risk_HIGH": [], "risk_MEDIUM": [], "risk_LOW": []}

    # Keep looping until no new facts are added in a full pass
    # This is called reaching the "fixed point"
    changed = True
    iteration = 0

    while changed:
        changed   = False
        iteration += 1

        for rule in RULES:
            # Skip rules already fired (avoid firing twice)
            if rule["name"] in fired_rules:
                continue

            # Check if ALL conditions of this rule are satisfied
            all_met = all(
                _check_condition(cond, working_memory)
                for cond in rule["conditions"]
            )

            if all_met:
                conclusion  = rule["conclusion"]
                confidence  = rule["confidence"]
                explanation = rule["explanation"]

                # Add the conclusion to working memory as a new fact
                # (This is what enables chaining — new facts trigger more rules)
                if conclusion not in working_memory:
                    working_memory[conclusion] = True
                    changed = True   # something new was added → loop again

                # Track risk conclusions separately to combine confidences
                if conclusion in risk_scores:
                    risk_scores[conclusion].append(confidence)

                fired_rules.append(rule["name"])
                explanations.append(explanation)
                trace.append(
                    f"  [{rule['name']}] fired → '{conclusion}' "
                    f"(confidence: {confidence})"
                )

    # --- Determine final risk level ---
    # If multiple rules concluded the same risk, combine their confidences.
    # We use: combined = 1 - product of (1 - c) for each c
    # This means two 0.8 rules together give: 1-(0.2*0.2) = 0.96
    # (stronger than either alone, but never reaches 1.0 by combining alone)

    def combine_confidences(scores):
        result = 1.0
        for s in scores:
            result *= (1.0 - s)
        return round(1.0 - result, 3)

    high_conf   = combine_confidences(risk_scores["risk_HIGH"])
    medium_conf = combine_confidences(risk_scores["risk_MEDIUM"])

    # Pick the highest combined confidence as the final risk level
    if high_conf > 0 and high_conf >= medium_conf:
        risk_level = "High"
        confidence = high_conf
    elif medium_conf > 0:
        risk_level = "Medium"
        confidence = medium_conf
    else:
        risk_level = "Low"
        confidence = round(1.0 - high_conf - medium_conf, 3)
        confidence = max(0.5, confidence)   # Low risk has baseline confidence

    return {
        "working_memory": working_memory,
        "fired_rules":    fired_rules,
        "trace":          trace,
        "risk_level":     risk_level,
        "confidence":     confidence,
        "explanations":   explanations,
        "iterations":     iteration,
    }


# -------------------------------------------------------------
# PART 4: BACKWARD CHAINING
# -------------------------------------------------------------
# Algorithm:
#   1. Start with a GOAL (e.g. "is this student HIGH risk?")
#   2. Find rules that could prove this goal
#   3. Check their conditions — are any facts missing?
#   4. If yes → ask the user a targeted question for that fact
#   5. Return the question to ask (or None if all facts are known)
#
# This is used for the "interactive question loop" requirement.
# Instead of asking every question upfront, the system only asks
# what it genuinely needs to reach a conclusion.

# Map: fact name → the question to ask the user if it's missing
QUESTIONS = {
    "days_until_deadline": "How many days until your next major deadline?",
    "stress_level":        "On a scale of 1–10, how stressed do you feel right now?",
    "confidence":          "On a scale of 1–10, how confident are you with the material?",
    "hours_available":     "How many hours can you study this week?",
    "missed_sessions":     "How many study sessions have you missed recently?",
    "gender":              "What is your gender? (female / male / other)",
}

def backward_chain(goal: str, working_memory: dict) -> dict:
    """
    Tries to prove a goal by working backward through the rules.

    Parameters:
        goal           — the conclusion we want to prove (e.g. "risk_HIGH")
        working_memory — facts we already know

    Returns:
        dict with keys:
            proved        — True if the goal is already provable
            missing_fact  — the key of the first missing fact needed
            question      — the question to ask the user to get that fact
            reasoning     — explanation of why we need this information
    """
    # Find all rules that could conclude our goal
    relevant_rules = [r for r in RULES if r["conclusion"] == goal]

    if not relevant_rules:
        return {
            "proved":       False,
            "missing_fact": None,
            "question":     None,
            "reasoning":    f"No rules exist to prove '{goal}'"
        }

    # Try each relevant rule — can any of them be satisfied?
    for rule in relevant_rules:
        missing = []

        for fact_key, operator, value in rule["conditions"]:
            if fact_key not in working_memory:
                missing.append(fact_key)

        # If no facts are missing for this rule, the goal is provable
        if not missing:
            all_met = all(
                _check_condition(cond, working_memory)
                for cond in rule["conditions"]
            )
            if all_met:
                return {
                    "proved":       True,
                    "missing_fact": None,
                    "question":     None,
                    "reasoning":    f"Goal '{goal}' proved via {rule['name']}"
                }

        # Otherwise, return a question for the first missing fact
        else:
            missing_fact = missing[0]
            question     = QUESTIONS.get(
                missing_fact,
                f"Can you tell me your {missing_fact}?"
            )
            return {
                "proved":       False,
                "missing_fact": missing_fact,
                "question":     question,
                "reasoning":    (
                    f"To check '{rule['name']}', I need to know "
                    f"'{missing_fact}' — asking the user."
                )
            }

    return {
        "proved":       False,
        "missing_fact": None,
        "question":     None,
        "reasoning":    f"Could not prove '{goal}' with available facts"
    }


# -------------------------------------------------------------
# PART 5: THE MAIN FUNCTION OTHER MODULES WILL CALL
# -------------------------------------------------------------

def run_expert_system(student_facts: dict,
                      use_backward_chain_for: str = "risk_HIGH") -> dict:
    """
    Full expert system run for one student.
    1. Uses backward chaining to check if any key fact is missing
    2. Runs forward chaining to derive all conclusions
    3. Returns everything needed for the UI to display

    Parameters:
        student_facts           — dict of student's known facts
        use_backward_chain_for  — goal to check via backward chaining

    Returns combined result dict
    """
    print("\n" + "=" * 55)
    print("  EXPERT SYSTEM — Rule-Based Reasoning")
    print("=" * 55)

    # --- Step A: Backward chaining check ---
    # Before we start, check if we're missing anything critical
    print(f"\n[Backward chaining] Checking what's needed to prove '{use_backward_chain_for}'...")
    bc_result = backward_chain(use_backward_chain_for, student_facts)
    print(f"  → {bc_result['reasoning']}")

    if not bc_result["proved"] and bc_result["question"]:
        print(f"  → Missing fact: '{bc_result['missing_fact']}'")
        print(f"  → Would ask: \"{bc_result['question']}\"")

    # --- Step B: Forward chaining ---
    print(f"\n[Forward chaining] Deriving conclusions from {len(student_facts)} known facts...")
    fc_result = forward_chain(student_facts)

    print(f"\n  Reasoning trace ({fc_result['iterations']} iteration(s)):")
    for step in fc_result["trace"]:
        print(step)

    print(f"\n  Rules fired: {len(fc_result['fired_rules'])}")
    print(f"  Final risk : {fc_result['risk_level']}")
    print(f"  Confidence : {fc_result['confidence']:.0%}")

    print("\n  Explanations:")
    for i, exp in enumerate(fc_result["explanations"], 1):
        print(f"    {i}. {exp}")

    # --- Step C: Build the final combined result ---
    return {
        "risk_level":        fc_result["risk_level"],
        "confidence":        fc_result["confidence"],
        "explanations":      fc_result["explanations"],
        "fired_rules":       fc_result["fired_rules"],
        "trace":             fc_result["trace"],
        "working_memory":    fc_result["working_memory"],
        "missing_question":  bc_result.get("question"),
        "missing_fact":      bc_result.get("missing_fact"),
    }


# -------------------------------------------------------------
# RUN THIS FILE DIRECTLY TO TEST EVERYTHING
# -------------------------------------------------------------
if __name__ == "__main__":

    # Test 1: High-risk student (all facts provided)
    print("\n" + "★" * 55)
    print("  TEST 1: High-risk student")
    print("★" * 55)

    high_risk_student = {
        "days_until_deadline": 2,
        "stress_level":        9,
        "confidence":          3,
        "hours_available":     4,
        "missed_sessions":     3,
        "gender":              "male",
    }
    result1 = run_expert_system(high_risk_student)

    # Test 2: Low-risk student
    print("\n" + "★" * 55)
    print("  TEST 2: Low-risk student")
    print("★" * 55)

    low_risk_student = {
        "days_until_deadline": 10,
        "stress_level":        3,
        "confidence":          8,
        "hours_available":     15,
        "missed_sessions":     0,
        "gender":              "female",
    }
    result2 = run_expert_system(low_risk_student)

    # Test 3: Missing information (backward chaining in action)
    print("\n" + "★" * 55)
    print("  TEST 3: Incomplete info — backward chaining asks a question")
    print("★" * 55)

    incomplete_student = {
        "days_until_deadline": 2,
        "stress_level":        8,
        # missed_sessions is intentionally missing
        # backward chaining should detect this and ask for it
    }
    result3 = run_expert_system(incomplete_student)

    if result3["missing_question"]:
        print(f"\n  System asks: \"{result3['missing_question']}\"")
        print("  (This is backward chaining — it only asks what it needs)")
