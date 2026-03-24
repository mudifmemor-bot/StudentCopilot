# =============================================================
# ui/main.py
# Student Success Copilot — Main Integration
#
# WHAT THIS FILE DOES:
#   This is the conductor. It:
#     1. Collects student input through an interactive question loop
#     2. Uses backward chaining to detect and fill missing information
#     3. Runs the rule-based expert system  (forward + backward chaining)
#     4. Runs the ML model                  (decision tree prediction)
#     5. Runs the search-based planner      (greedy vs A* comparison)
#     6. Combines all outputs into one unified response
#     7. Demonstrates two full scenarios:   normal + at-risk student
#
# HOW TO RUN:
#   python ui/main.py
#
# PROJECT STRUCTURE (all files needed):
#   student_copilot/
#     ml/risk_model.py
#     rules/expert.py
#     planner/search.py
#     ui/main.py          ← this file
# =============================================================

import sys
import os
import time

# --- Make sure Python can find our other modules ---
# This adds the parent folder to the path so we can import
# from ml/, rules/, and planner/ regardless of where you run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.risk_model    import train_model, predict_risk
from rules.expert     import run_expert_system, backward_chain, QUESTIONS
from planner.search   import build_schedule, Task


# =============================================================
# PART 1: INPUT COLLECTION WITH BACKWARD CHAINING LOOP
# =============================================================
# The system doesn't ask every question upfront.
# It asks the minimum needed — and uses backward chaining
# to decide if a follow-up question is required.

def ask(prompt: str, cast=str, valid=None):
    """
    Asks the user a question and validates the answer.

    Parameters:
        prompt — the question to display
        cast   — type to convert to (int, float, str)
        valid  — optional list of valid values
    """
    while True:
        try:
            raw = input(f"\n  {prompt}\n  → ").strip()
            value = cast(raw)
            if valid and value not in valid:
                print(f"  Please choose from: {valid}")
                continue
            return value
        except (ValueError, TypeError):
            print(f"  Please enter a valid {cast.__name__}.")


def collect_student_info() -> dict:
    """
    Interactively collects a student's information.
    Uses backward chaining to ask follow-up questions
    only when needed — satisfying the coursework requirement
    for an interactive question loop.

    Returns:
        dict of student facts ready for the AI components
    """
    print("\n" + "=" * 55)
    print("  STUDENT SUCCESS COPILOT")
    print("  Personalised study planner + risk detector")
    print("=" * 55)

    print("\n  Let's build your personalised study plan.")
    print("  Answer the questions below (press Enter after each).\n")

    # --- Core questions (always asked) ---
    name = ask("What is your name?", str)

    gender = ask(
        "What is your gender? (female / male / other)",
        str,
        valid=["female", "male", "other"]
    )

    stress = ask(
        "On a scale of 1–10, how stressed do you feel right now?\n"
        "  (1 = completely calm, 10 = overwhelmed)",
        int
    )

    confidence = ask(
        "On a scale of 1–10, how confident are you with your current material?\n"
        "  (1 = lost, 10 = fully on top of it)",
        int
    )

    days = ask(
        "How many days until your next major deadline?",
        int
    )

    hours = ask(
        "How many hours can you study this week in total?",
        int
    )

    missed = ask(
        "How many study sessions have you missed recently?",
        int
    )

    # Build the initial facts dictionary
    student_facts = {
        "name":                name,
        "gender":              gender,
        "stress_level":        stress,
        "confidence":          confidence,
        "days_until_deadline": days,
        "hours_available":     hours,
        "missed_sessions":     missed,
    }

    # --- Backward chaining: do we need more info? ---
    # Check if we can already conclude risk_HIGH.
    # If a critical fact is still missing, ask for it now.
    print("\n  [Backward chaining] Checking if any critical info is missing...")
    bc = backward_chain("risk_HIGH", student_facts)

    if not bc["proved"] and bc["question"] and bc["missing_fact"]:
        missing = bc["missing_fact"]
        # Only ask if it's genuinely not in our facts yet
        if missing not in student_facts:
            print(f"  → To assess your risk accurately, I need one more thing.")
            extra_value = ask(bc["question"])
            # Try to convert to int if possible (most facts are numeric)
            try:
                student_facts[missing] = int(extra_value)
            except ValueError:
                student_facts[missing] = extra_value

    # --- Collect tasks ---
    print("\n  Now let's add the tasks you need to complete.")
    tasks = collect_tasks()
    student_facts["tasks"] = tasks

    # --- Collect weekly availability ---
    availability = collect_availability(hours)
    student_facts["availability"] = availability

    return student_facts


def collect_tasks() -> list:
    """Collects the student's task list interactively."""
    tasks = []
    day_names = ["Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday", "Sunday"]

    while True:
        print(f"\n  Task {len(tasks) + 1}:")
        task_name = ask("Task name (e.g. 'AI Coursework', or 'done' to finish):", str)

        if task_name.lower() == "done":
            if not tasks:
                print("  Please add at least one task.")
                continue
            break

        task_days  = ask(f"  Days until this task is due (1–14):", int)
        task_hours = ask(f"  Estimated hours needed to complete it:", float)
        task_diff  = ask(f"  Difficulty (1=easy, 2=medium, 3=hard):", int, valid=[1, 2, 3])
        task_subj  = ask(f"  Subject (e.g. AI, Maths, English):", str)

        tasks.append(Task(
            name=task_name,
            days_until_due=task_days,
            hours_needed=task_hours,
            difficulty=task_diff,
            subject=task_subj,
        ))
        print(f"  ✓ Added: {task_name}")

    return tasks


def collect_availability(total_hours: int) -> dict:
    """
    Builds a weekly availability map from the student's input.
    Distributes the total hours across days they are free.
    """
    print(f"\n  You have {total_hours} study hours this week.")
    print("  Which days are you available? (enter hours per day, 0 = not available)")

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    availability = {}
    total_allocated = 0

    for day in days:
        if total_allocated >= total_hours:
            availability[day] = 0
            continue
        remaining = total_hours - total_allocated
        h = ask(f"  {day} (max {remaining}h remaining):", float)
        h = min(h, remaining)   # never exceed total
        availability[day] = h
        total_allocated += h
        if total_allocated >= total_hours:
            print(f"  All {total_hours} hours allocated.")
            break

    return availability


# =============================================================
# PART 2: RUN ALL THREE AI COMPONENTS
# =============================================================

def run_all_components(student_facts: dict, ml_model) -> dict:
    """
    Runs all three AI components on the student's data
    and returns their combined outputs.

    Parameters:
        student_facts — dict from collect_student_info()
        ml_model      — trained Decision Tree from train_model()

    Returns:
        dict with keys: expert_result, ml_result, schedule_result
    """
    print("\n" + "=" * 55)
    print("  RUNNING AI COMPONENTS")
    print("=" * 55)

    # --- Component 1: Rule-based expert system ---
    print("\n[1/3] Rule-based expert system...")
    # Pass only the numeric/string facts (not tasks/availability)
    facts_for_rules = {k: v for k, v in student_facts.items()
                       if k not in ("tasks", "availability", "name")}
    expert_result = run_expert_system(facts_for_rules)

    # --- Component 2: ML model ---
    print("\n[2/3] ML risk prediction...")
    ml_result = predict_risk(ml_model, student_facts)
    print(f"  ML prediction : {ml_result['risk']} risk")
    print(f"  ML confidence : {ml_result['confidence_pct']}%")
    print(f"  ML explanation: {ml_result['explanation']}")

    # --- Combine risk signals ---
    # Both the rule engine and the ML model produce a risk level.
    # We take the more cautious (higher) of the two — it's better
    # to flag a student as at-risk and be wrong than to miss them.
    risk_order = {"Low": 0, "Medium": 1, "High": 2}
    rule_risk  = expert_result["risk_level"]
    ml_risk    = ml_result["risk"]

    if risk_order[ml_risk] >= risk_order[rule_risk]:
        final_risk = ml_risk
        risk_source = "ML model"
    else:
        final_risk = rule_risk
        risk_source = "rule engine"

    print(f"\n  Combined risk  : {final_risk} (from {risk_source} — most cautious signal)")

    # --- Component 3: Search planner ---
    print("\n[3/3] Building study schedule (Greedy vs A*)...")
    tasks        = student_facts.get("tasks", [])
    availability = student_facts.get("availability", {})

    schedule_result = build_schedule(tasks, availability, risk_level=final_risk)

    return {
        "expert_result":   expert_result,
        "ml_result":       ml_result,
        "schedule_result": schedule_result,
        "final_risk":      final_risk,
        "risk_source":     risk_source,
    }


# =============================================================
# PART 3: FINAL OUTPUT — unified, readable, actionable
# =============================================================

def print_final_output(student_facts: dict, results: dict) -> None:
    """
    Prints the complete, unified output for the student.
    This is what they actually see — the payoff of all three components.
    """
    name        = student_facts.get("name", "Student")
    final_risk  = results["final_risk"]
    expert      = results["expert_result"]
    ml          = results["ml_result"]
    schedule    = results["schedule_result"]

    # Risk level → emoji indicator
    risk_icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    icon = risk_icons.get(final_risk, "⚪")

    print("\n\n" + "★" * 55)
    print(f"  YOUR PERSONALISED STUDY PLAN, {name.upper()}")
    print("★" * 55)

    # --- 1. Risk level ---
    print(f"\n  {icon}  RISK LEVEL: {final_risk.upper()}")
    print(f"      Rule engine: {expert['risk_level']} "
          f"({expert['confidence']:.0%} confidence)")
    print(f"      ML model   : {ml['risk']} "
          f"({ml['confidence_pct']}% confidence)")

    # --- 2. Why this risk ---
    print(f"\n  WHY THIS RISK LEVEL:")
    for i, explanation in enumerate(expert["explanations"][:5], 1):
        print(f"    {i}. {explanation}")
    if ml["factors"]:
        print(f"    ML also flagged: {'; '.join(ml['factors'])}")

    # --- 3. Study schedule ---
    print(f"\n  YOUR WEEKLY SCHEDULE (A* optimised):")
    best_schedule = schedule["astar"]["scheduled"]

    if not best_schedule:
        print("    No tasks could be scheduled. Consider freeing up more time.")
    else:
        days_seen = {}
        for item in best_schedule:
            days_seen.setdefault(item.slot.day_name, []).append(item)

        for day_name, items in sorted(days_seen.items(),
                                      key=lambda x: x[1][0].slot.day):
            print(f"\n    {day_name}:")
            for item in items:
                print(f"      {item.slot.start_hour:02d}:00 — "
                      f"{item.hours_used:.1f}h — {item.task.name} "
                      f"[due in {item.task.days_until_due}d]")

    if schedule["astar"]["unscheduled"]:
        print(f"\n  ⚠  COULD NOT FIT (not enough time):")
        for t in schedule["astar"]["unscheduled"]:
            print(f"    - {t.name} (needs {t.hours_needed}h)")

    # --- 4. Search comparison ---
    g = schedule["greedy"]
    a = schedule["astar"]
    print(f"\n  SEARCH STRATEGY COMPARISON:")
    print(f"    {'Strategy':<12} {'Quality':>9} {'Time':>10}")
    print(f"    {'─'*35}")
    print(f"    {'Greedy':<12} {g['quality_score']:>9} {g['time_ms']:>8}ms")
    print(f"    {'A*':<12} {a['quality_score']:>9} {a['time_ms']:>8}ms")
    print(f"    Winner: {schedule['winner']} "
          f"(A* uses heuristic scoring — smarter for at-risk students)")

    # --- 5. Recommendations ---
    print(f"\n  RECOMMENDATIONS:")
    if final_risk == "High":
        print("    1. Focus exclusively on your most urgent task first.")
        print("    2. Contact your lecturer today — explain your situation.")
        print("    3. Drop non-essential activities this week.")
        print("    4. Study in short focused blocks (25 min on, 5 min off).")
        print("    5. Attend every remaining session — attendance affects risk.")
    elif final_risk == "Medium":
        print("    1. Stick to the schedule above — don't skip planned sessions.")
        print("    2. Tackle difficult tasks when your energy is highest.")
        print("    3. Review your confidence gaps before the deadline.")
        print("    4. Reach out for help if stress increases.")
    else:
        print("    1. You are in good shape — maintain your current pace.")
        print("    2. Use the schedule to stay organised.")
        print("    3. Review material regularly rather than cramming.")

    # --- 6. AI reasoning trace (for transparency) ---
    print(f"\n  AI REASONING TRACE (how we reached this conclusion):")
    for step in expert["trace"][:6]:
        print(f"    {step}")
    if len(expert["trace"]) > 6:
        print(f"    ... and {len(expert['trace']) - 6} more rules")

    print("\n" + "★" * 55)
    print("  End of report. Good luck with your studies!")
    print("★" * 55 + "\n")


# =============================================================
# PART 4: DEMO SCENARIOS (for the coursework video)
# =============================================================
# The coursework requires two demo scenarios:
#   1. A "normal" student (low risk)
#   2. An "at-risk" student (high risk)
# These run automatically without user input — perfect for the video.

def run_demo_scenario(scenario_name: str,
                      student_facts: dict,
                      ml_model) -> None:
    """Runs one pre-defined demo scenario end to end."""
    print("\n\n" + "=" * 55)
    print(f"  DEMO SCENARIO: {scenario_name}")
    print("=" * 55)

    results = run_all_components(student_facts, ml_model)
    print_final_output(student_facts, results)


def run_demos(ml_model) -> None:
    """Runs both coursework demo scenarios automatically."""

    # --- Scenario 1: Normal student ---
    normal_student = {
        "name":                "Alex",
        "gender":              "male",
        "stress_level":        3,
        "confidence":          8,
        "days_until_deadline": 10,
        "hours_available":     15,
        "missed_sessions":     0,
        "tasks": [
            Task("AI Assignment",  days_until_due=10, hours_needed=4,
                 difficulty=3, subject="AI"),
            Task("Maths Homework", days_until_due=7,  hours_needed=2,
                 difficulty=2, subject="Maths"),
            Task("Essay",          days_until_due=12, hours_needed=3,
                 difficulty=1, subject="English"),
        ],
        "availability": {
            "Monday":    3,
            "Tuesday":   2,
            "Wednesday": 3,
            "Thursday":  3,
            "Friday":    2,
            "Saturday":  2,
        },
    }

    # --- Scenario 2: At-risk student ---
    atrisk_student = {
        "name":                "Sam",
        "gender":              "female",
        "stress_level":        9,
        "confidence":          2,
        "days_until_deadline": 2,
        "hours_available":     5,
        "missed_sessions":     3,
        "tasks": [
            Task("AI Coursework",  days_until_due=2, hours_needed=5,
                 difficulty=3, subject="AI"),
            Task("Maths Exam",     days_until_due=2, hours_needed=3,
                 difficulty=3, subject="Maths"),
            Task("Lab Report",     days_until_due=3, hours_needed=2,
                 difficulty=2, subject="Science"),
        ],
        "availability": {
            "Monday":  2,
            "Tuesday": 3,
        },
    }

    run_demo_scenario("Normal student (Alex)", normal_student, ml_model)
    run_demo_scenario("At-risk student (Sam)",  atrisk_student, ml_model)


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    print("\n" + "=" * 55)
    print("  STUDENT SUCCESS COPILOT — Starting up")
    print("=" * 55)

    # Train the ML model once at startup
    print("\n  Training ML model...")
    ml_model, _ = train_model()
    print("  ML model ready.")

    # Ask: demo mode or live interactive mode?
    print("\n  Run mode:")
    print("    1 — Demo mode   (two preset scenarios, great for video)")
    print("    2 — Interactive (enter your own data)")
    mode = input("\n  Choose (1 or 2): ").strip()

    if mode == "1":
        run_demos(ml_model)

    else:
        # Interactive mode — collect real input then run
        student_facts = collect_student_info()
        results       = run_all_components(student_facts, ml_model)
        print_final_output(student_facts, results)

    print("\n  Done. Check the output above for your study plan.\n")


if __name__ == "__main__":
    main()
