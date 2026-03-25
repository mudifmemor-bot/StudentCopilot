# =============================================================
# extensions/fuzzy_controller.py
# Extension 1: Fuzzy Logic Stress/Workload Controller
#
# CONCEPT:
#   Normal (crisp) logic: stress >= 7 → high stress (binary)
#   Fuzzy logic: stress = 6 → 40% high, 60% medium (gradual)
#
# THREE STEPS:
#   1. FUZZIFY   — turn crisp numbers into membership degrees
#   2. INFER     — apply fuzzy IF-THEN rules
#   3. DEFUZZIFY — turn fuzzy output back into one crisp number
#
# INPUTS:  stress (1-10), free_time (hours/week), task_difficulty (1-3)
# OUTPUTS: recommended_intensity (1-10), plan_adjustment (text)
# =============================================================


# -------------------------------------------------------------
# STEP 1: MEMBERSHIP FUNCTIONS
# -------------------------------------------------------------
# A membership function takes a crisp value and returns how
# much it "belongs" to a fuzzy set (0.0 = not at all, 1.0 = fully).
#
# We use triangular functions — simple, readable, effective.
# triangle(x, a, b, c):
#   - rises from 0→1 between a and b
#   - falls from 1→0 between b and c
#   - zero outside [a, c]

def triangle(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.
    Returns how much x belongs to the fuzzy set defined by (a, b, c).
    b is the peak (membership=1.0), a and c are the feet (membership=0.0).
    """
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function — flat top between b and c.
    Used for extreme ends (e.g. "very low" has full membership from 1–2).
    """
    if x <= a or x >= d:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a)
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


# --- Stress membership functions (scale 1–10) ---

def stress_low(s):
    """Fully low at 1-2, fades to 0 by 5."""
    return trapezoid(s, 0.5, 1.0, 2.5, 5.0)

def stress_medium(s):
    """Peaks at 5, zero at 1 and 9."""
    return triangle(s, 1.0, 5.0, 9.0)

def stress_high(s):
    """Starts rising at 5, fully high at 8-10."""
    return trapezoid(s, 5.0, 7.5, 10.0, 10.5)


# --- Free time membership functions (hours/week, 0–20) ---

def time_scarce(t):
    """Fully scarce at 0-3h, fades to 0 by 8h."""
    return trapezoid(t, -0.5, 0.0, 3.0, 8.0)

def time_moderate(t):
    """Peaks at 10h."""
    return triangle(t, 3.0, 10.0, 17.0)

def time_ample(t):
    """Fully ample at 15h+."""
    return trapezoid(t, 10.0, 15.0, 20.0, 20.5)


# --- Task difficulty membership functions (scale 1–3) ---

def diff_easy(d):
    """Fully easy at 1, fades by 2."""
    return trapezoid(d, 0.5, 1.0, 1.3, 2.0)

def diff_medium(d):
    """Peaks at 2."""
    return triangle(d, 1.0, 2.0, 3.0)

def diff_hard(d):
    """Fully hard at 2.7+."""
    return trapezoid(d, 2.0, 2.7, 3.0, 3.5)


# --- Intensity OUTPUT membership functions (scale 1–10) ---
# These define what the output "means" before defuzzification.

def intensity_light(i):
    """Light intensity: 1-3."""
    return trapezoid(i, 0.5, 1.0, 2.5, 4.5)

def intensity_moderate(i):
    """Moderate intensity: peaks at 5."""
    return triangle(i, 2.5, 5.0, 7.5)

def intensity_intense(i):
    """Intense: 7-10."""
    return trapezoid(i, 5.5, 7.5, 10.0, 10.5)


# -------------------------------------------------------------
# STEP 2: FUZZIFICATION
# -------------------------------------------------------------
# Takes crisp input values and converts them to membership degrees.

def fuzzify(stress: float, free_time: float, avg_difficulty: float) -> dict:
    """
    Converts crisp inputs into fuzzy membership degrees.

    Returns a dict of all membership values — one per fuzzy set.
    """
    return {
        # Stress memberships
        "stress_low":    stress_low(stress),
        "stress_medium": stress_medium(stress),
        "stress_high":   stress_high(stress),

        # Free time memberships
        "time_scarce":   time_scarce(free_time),
        "time_moderate": time_moderate(free_time),
        "time_ample":    time_ample(free_time),

        # Difficulty memberships
        "diff_easy":     diff_easy(avg_difficulty),
        "diff_medium":   diff_medium(avg_difficulty),
        "diff_hard":     diff_hard(avg_difficulty),
    }


# -------------------------------------------------------------
# STEP 3: FUZZY INFERENCE (IF-THEN rules)
# -------------------------------------------------------------
# Each rule has:
#   - antecedent: the IF part (combined with min() = fuzzy AND)
#   - consequent: the output fuzzy set this rule activates
#   - strength:   how strongly this rule fires (= antecedent value)
#
# min(a, b) is fuzzy AND — a chain is only as strong as its weakest link.
# max(a, b) is fuzzy OR  — at least one must be true.

def infer(memberships: dict) -> dict:
    """
    Applies all fuzzy rules and returns the activation
    strength for each output fuzzy set.

    Uses Mamdani inference (min for AND, max to aggregate).
    """
    m = memberships  # shorthand

    # Rule activations: each rule fires to a strength between 0 and 1.
    # The strength clips the output membership function at that level.

    rules = [
        # Rule 1: IF stress_high AND time_scarce → intensity_intense
        ("intense", min(m["stress_high"], m["time_scarce"])),

        # Rule 2: IF stress_high AND time_moderate → intensity_intense
        ("intense", min(m["stress_high"], m["time_moderate"])),

        # Rule 3: IF stress_medium AND time_scarce → intensity_moderate
        ("moderate", min(m["stress_medium"], m["time_scarce"])),

        # Rule 4: IF stress_medium AND time_moderate → intensity_moderate
        ("moderate", min(m["stress_medium"], m["time_moderate"])),

        # Rule 5: IF stress_low AND time_ample → intensity_light
        ("light", min(m["stress_low"], m["time_ample"])),

        # Rule 6: IF stress_low AND time_moderate → intensity_light
        ("light", min(m["stress_low"], m["time_moderate"])),

        # Rule 7: IF diff_hard AND time_scarce → intensity_intense
        ("intense", min(m["diff_hard"], m["time_scarce"])),

        # Rule 8: IF diff_easy AND stress_low → intensity_light
        ("light", min(m["diff_easy"], m["stress_low"])),

        # Rule 9: IF diff_hard AND stress_high → intensity_intense (max boost)
        ("intense", min(m["diff_hard"], m["stress_high"])),
    ]

    # Aggregate: for each output set, take the MAXIMUM activation
    # (multiple rules can activate the same output — we take the strongest)
    aggregated = {"light": 0.0, "moderate": 0.0, "intense": 0.0}
    for output_set, strength in rules:
        aggregated[output_set] = max(aggregated[output_set], strength)

    return aggregated


# -------------------------------------------------------------
# STEP 4: DEFUZZIFICATION (centroid method)
# -------------------------------------------------------------
# We have fuzzy output activations — now we need ONE crisp number.
# Centroid method: sample the output space, weight each point by
# the clipped membership, find the "centre of mass".
#
# Imagine a shape made of the activated output functions.
# The centroid is the balance point of that shape.

def defuzzify(aggregated: dict, resolution: int = 100) -> float:
    """
    Converts aggregated fuzzy output to a single crisp intensity value.
    Uses the centroid (centre of mass) defuzzification method.

    Parameters:
        aggregated  — dict of {set_name: activation_strength}
        resolution  — number of sample points (more = more accurate)

    Returns:
        crisp intensity value on a 1–10 scale
    """
    numerator   = 0.0
    denominator = 0.0

    # Sample the output universe at `resolution` evenly-spaced points
    for i in range(resolution):
        x = 1.0 + (9.0 * i / (resolution - 1))  # x from 1 to 10

        # At this point x, what is the clipped membership for each output set?
        # Clipping: cap the membership function at the activation strength
        light_mem    = min(intensity_light(x),    aggregated["light"])
        moderate_mem = min(intensity_moderate(x), aggregated["moderate"])
        intense_mem  = min(intensity_intense(x),  aggregated["intense"])

        # Take the maximum across all output sets at this point (aggregation)
        combined = max(light_mem, moderate_mem, intense_mem)

        numerator   += x * combined
        denominator += combined

    if denominator == 0:
        return 5.0  # neutral fallback if nothing fired

    return round(numerator / denominator, 2)


# -------------------------------------------------------------
# STEP 5: PLAN ADJUSTMENT (qualitative output)
# -------------------------------------------------------------
# Based on the crisp intensity, generate a human-readable
# recommendation for how to adjust the study plan.

def plan_adjustment(intensity: float,
                    stress: float,
                    free_time: float) -> dict:
    """
    Translates crisp intensity into actionable advice.

    Returns a dict with intensity_label, advice, and session_length.
    """
    if intensity >= 7.0:
        label        = "Intense"
        session_len  = 25   # Pomodoro blocks — shorter, more focused
        advice = [
            "Your workload demands intense focus this week.",
            "Study in 25-minute focused blocks with 5-minute breaks.",
            "Prioritise your hardest task every morning when energy is highest.",
            "Cut optional activities — protect every available study hour.",
            "Consider asking for an extension if workload is unmanageable.",
        ]
    elif intensity >= 4.5:
        label        = "Moderate"
        session_len  = 45
        advice = [
            "Your workload is manageable with consistent effort.",
            "Study in 45-minute sessions — enough depth without burnout.",
            "Balance difficult subjects with easier review tasks.",
            "Take one full rest day to avoid accumulated fatigue.",
        ]
    else:
        label        = "Light"
        session_len  = 60
        advice = [
            "You have comfortable breathing room this week.",
            "Use longer 60-minute deep-work sessions to make real progress.",
            "Focus on understanding deeply, not just completing tasks.",
            "This is a good week to review past material and fill gaps.",
        ]

    return {
        "intensity_label": label,
        "session_length":  session_len,
        "advice":          advice,
    }


# -------------------------------------------------------------
# MAIN FUNCTION — full fuzzy controller pipeline
# -------------------------------------------------------------

def run_fuzzy_controller(stress: float,
                         free_time: float,
                         avg_difficulty: float) -> dict:
    """
    Full fuzzy logic pipeline: fuzzify → infer → defuzzify → advise.

    Parameters:
        stress          — student's stress level (1–10)
        free_time       — available study hours this week
        avg_difficulty  — average task difficulty (1–3)

    Returns:
        dict with all results and explanation
    """
    print("\n" + "=" * 55)
    print("  FUZZY LOGIC — Workload Intensity Controller")
    print("=" * 55)

    # Step 1: Fuzzify
    memberships = fuzzify(stress, free_time, avg_difficulty)
    print(f"\n  [1] Fuzzification (stress={stress}, "
          f"free_time={free_time}h, difficulty={avg_difficulty}):")
    print(f"      stress  → low:{memberships['stress_low']:.2f}  "
          f"medium:{memberships['stress_medium']:.2f}  "
          f"high:{memberships['stress_high']:.2f}")
    print(f"      time    → scarce:{memberships['time_scarce']:.2f}  "
          f"moderate:{memberships['time_moderate']:.2f}  "
          f"ample:{memberships['time_ample']:.2f}")
    print(f"      diff    → easy:{memberships['diff_easy']:.2f}  "
          f"medium:{memberships['diff_medium']:.2f}  "
          f"hard:{memberships['diff_hard']:.2f}")

    # Step 2: Infer
    aggregated = infer(memberships)
    print(f"\n  [2] Fuzzy inference — output activations:")
    print(f"      light:{aggregated['light']:.2f}  "
          f"moderate:{aggregated['moderate']:.2f}  "
          f"intense:{aggregated['intense']:.2f}")

    # Step 3: Defuzzify
    intensity = defuzzify(aggregated)
    print(f"\n  [3] Defuzzification (centroid method):")
    print(f"      Recommended intensity = {intensity}/10")

    # Step 4: Plan adjustment
    adjustment = plan_adjustment(intensity, stress, free_time)
    print(f"\n  [4] Plan adjustment:")
    print(f"      Intensity level : {adjustment['intensity_label']}")
    print(f"      Session length  : {adjustment['session_length']} minutes")
    print(f"      Advice:")
    for tip in adjustment["advice"]:
        print(f"        - {tip}")

    return {
        "stress":          stress,
        "free_time":       free_time,
        "avg_difficulty":  avg_difficulty,
        "memberships":     memberships,
        "aggregated":      aggregated,
        "intensity":       intensity,
        "adjustment":      adjustment,
    }


# -------------------------------------------------------------
# TEST
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n★ Test 1: High-stress, scarce time, hard tasks")
    run_fuzzy_controller(stress=9, free_time=4, avg_difficulty=3)

    print("\n★ Test 2: Medium stress, moderate time")
    run_fuzzy_controller(stress=5, free_time=10, avg_difficulty=2)

    print("\n★ Test 3: Low stress, plenty of time, easy tasks")
    run_fuzzy_controller(stress=2, free_time=18, avg_difficulty=1)
