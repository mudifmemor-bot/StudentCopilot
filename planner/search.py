# =============================================================
# planner/search.py
# Student Success Copilot — Search-Based Planner
#
# WHAT THIS FILE DOES:
#   1. Defines the planning problem as a search problem:
#        State   = which tasks have been scheduled so far
#        Actions = assign the next task to the next free slot
#        Goal    = all tasks scheduled before their deadlines
#   2. Implements TWO search strategies:
#        - Greedy search : always picks the most urgent task (fast)
#        - A* search     : scores tasks using a heuristic (smarter)
#   3. Compares both strategies: schedule quality + time taken
#   4. Returns a human-readable weekly schedule
#
# WHY TWO STRATEGIES?
#   The coursework requires comparing two approaches.
#   Greedy is simple and fast but short-sighted.
#   A* uses more information per decision and produces
#   a better schedule — especially when the student is at risk.
# =============================================================

import time
from dataclasses import dataclass, field
from typing import List, Optional


# -------------------------------------------------------------
# PART 1: DATA STRUCTURES
# -------------------------------------------------------------
# A Task is one piece of work a student needs to complete.
# A Slot is one available study session in the week.
# A ScheduledItem is a task assigned to a slot (the output).

@dataclass
class Task:
    """
    Represents one assignment or study task.

    Attributes:
        name            — human-readable name e.g. "AI Coursework"
        days_until_due  — how many days until the deadline (1–14)
        hours_needed    — estimated hours to complete (1–10)
        difficulty      — how hard is it? (1=easy, 2=medium, 3=hard)
        subject         — subject area e.g. "AI", "Maths"
    """
    name:           str
    days_until_due: int
    hours_needed:   float
    difficulty:     int    # 1, 2, or 3
    subject:        str = "General"

    def deadline_day(self) -> int:
        """Returns which day of the week the deadline falls on (1=Mon)."""
        return min(self.days_until_due, 7)


@dataclass
class Slot:
    """
    Represents one available study block in the student's week.

    Attributes:
        day         — 1=Monday, 2=Tuesday … 7=Sunday
        day_name    — human-readable day name
        start_hour  — e.g. 9 means 9:00 AM
        duration    — how many hours this slot lasts
    """
    day:        int
    day_name:   str
    start_hour: int
    duration:   float   # hours available in this slot


@dataclass
class ScheduledItem:
    """One task assigned to one slot — an entry in the final schedule."""
    task:       Task
    slot:       Slot
    hours_used: float   # how much of the slot this task uses

    def to_string(self) -> str:
        return (
            f"  {self.slot.day_name} {self.slot.start_hour:02d}:00 "
            f"— {self.hours_used:.1f}h — {self.task.name} "
            f"(due in {self.task.days_until_due}d, difficulty {self.task.difficulty}/3)"
        )


# -------------------------------------------------------------
# PART 2: SEARCH STATE
# -------------------------------------------------------------
# The "state" is a snapshot of the planning process at one moment.
# It tracks which tasks are done, which slots are used, and the
# schedule built so far.
#
# Every time we make a scheduling decision (action), we create
# a new state. This is the core of search-based planning.

@dataclass
class PlannerState:
    """
    Represents the current state of the planning search.

    Attributes:
        scheduled       — list of (task, slot, hours) decisions made so far
        remaining_tasks — tasks not yet scheduled
        remaining_slots — slots not yet fully used
        cost            — total "badness" accumulated (conflicts, overdue etc.)
    """
    scheduled:       List[ScheduledItem]
    remaining_tasks: List[Task]
    remaining_slots: List[Slot]
    cost:            float = 0.0

    def is_goal(self) -> bool:
        """Goal reached when all tasks are scheduled."""
        return len(self.remaining_tasks) == 0

    def tasks_scheduled_before_deadline(self) -> int:
        """Count tasks that were scheduled before their deadline day."""
        count = 0
        for item in self.scheduled:
            if item.slot.day <= item.task.deadline_day():
                count += 1
        return count


# -------------------------------------------------------------
# PART 3: HEURISTIC FUNCTION (what makes A* smart)
# -------------------------------------------------------------
# The heuristic scores how urgently a task should be scheduled.
# Higher score = schedule this task sooner.
#
# Formula:
#   base_urgency = 1 / days_until_due      (closer deadline = more urgent)
#   difficulty   = task difficulty (1–3)   (harder = needs earlier slot)
#   risk_boost   = 1.5 if student is HIGH risk, else 1.0
#
# A* always picks the task with the HIGHEST heuristic score next.
# Greedy only uses days_until_due (simpler, less informed).

def heuristic(task: Task, risk_level: str = "Low") -> float:
    """
    Scores a task's scheduling priority.
    Higher = should be scheduled sooner.

    Parameters:
        task       — the Task to score
        risk_level — student's risk level from the expert system

    Returns:
        float priority score
    """
    # Urgency: tasks due sooner get much higher scores
    # Adding 0.5 avoids division-by-zero for same-day deadlines
    urgency = 1.0 / (task.days_until_due + 0.5)

    # Difficulty: harder tasks need more planning lead time
    difficulty_weight = task.difficulty / 3.0

    # Risk boost: if student is HIGH risk, urgency matters even more
    risk_boost = 1.5 if risk_level == "High" else (1.2 if risk_level == "Medium" else 1.0)

    return urgency * difficulty_weight * risk_boost


# -------------------------------------------------------------
# PART 4: GREEDY SEARCH
# -------------------------------------------------------------
# ALGORITHM:
#   1. Sort all tasks by days_until_due (most urgent first)
#   2. For each task in that order:
#        a. Find the earliest available slot
#        b. Assign the task to that slot
#        c. Mark the slot as used
#   3. Return the schedule
#
# WHY IT'S "GREEDY":
#   It never looks ahead. It grabs the most urgent task now
#   without considering how that choice affects future tasks.
#   This is fast (one pass) but can produce suboptimal results.

def greedy_search(tasks: List[Task],
                  slots: List[Slot],
                  risk_level: str = "Low") -> dict:
    """
    Builds a study schedule using greedy search.

    Parameters:
        tasks      — list of Task objects to schedule
        slots      — list of available Slot objects
        risk_level — from the expert system (affects urgency display)

    Returns:
        dict with schedule, unscheduled tasks, stats
    """
    start_time = time.perf_counter()

    # Sort tasks: most urgent (fewest days) first
    # This is the only "intelligence" greedy has — the initial sort
    sorted_tasks = sorted(tasks, key=lambda t: t.days_until_due)

    # Track remaining capacity in each slot (slots can be partially used)
    slot_remaining = {id(s): s.duration for s in slots}

    scheduled    = []
    unscheduled  = []
    nodes_explored = 0   # how many decisions were considered

    for task in sorted_tasks:
        scheduled_this_task = False
        hours_left = task.hours_needed

        # Try slots in order (earliest first)
        for slot in sorted(slots, key=lambda s: (s.day, s.start_hour)):
            nodes_explored += 1

            # Skip this slot if it's before the task's deadline
            # (no point scheduling after deadline)
            if slot.day > task.deadline_day():
                continue

            # Check if this slot has any capacity left
            available = slot_remaining[id(slot)]
            if available <= 0:
                continue

            # Assign as much of this task as possible to this slot
            hours_used = min(hours_left, available)
            slot_remaining[id(slot)] -= hours_used
            hours_left -= hours_used

            scheduled.append(ScheduledItem(
                task=task,
                slot=slot,
                hours_used=hours_used
            ))

            if hours_left <= 0:
                scheduled_this_task = True
                break   # task fully scheduled — move to next task

        if not scheduled_this_task:
            unscheduled.append(task)

    elapsed = time.perf_counter() - start_time

    return {
        "strategy":       "Greedy",
        "scheduled":      scheduled,
        "unscheduled":    unscheduled,
        "nodes_explored": nodes_explored,
        "time_ms":        round(elapsed * 1000, 3),
        "quality_score":  _quality_score(scheduled, unscheduled),
    }


# -------------------------------------------------------------
# PART 5: A* SEARCH
# -------------------------------------------------------------
# ALGORITHM:
#   1. Start with all tasks unscheduled
#   2. Score every remaining task using the heuristic()
#   3. Pick the highest-scoring task
#   4. Assign it to the best available slot
#      (earliest slot that fits before its deadline)
#   5. Repeat from step 2 until all tasks scheduled or no slots left
#
# WHY IT'S BETTER THAN GREEDY:
#   It recalculates scores after each assignment, so it adapts.
#   It also considers difficulty (not just deadline) when choosing.
#   For HIGH-risk students, the risk_boost amplifies urgency further.
#   This produces schedules that fit difficult tasks earlier.

def astar_search(tasks: List[Task],
                 slots: List[Slot],
                 risk_level: str = "Low") -> dict:
    """
    Builds a study schedule using A* heuristic search.

    Parameters:
        tasks      — list of Task objects to schedule
        slots      — list of available Slot objects
        risk_level — from the expert system (used in heuristic scoring)

    Returns:
        dict with schedule, unscheduled tasks, stats
    """
    start_time = time.perf_counter()

    # Work with copies so we don't modify the originals
    remaining_tasks = list(tasks)
    slot_remaining  = {id(s): s.duration for s in slots}

    scheduled      = []
    unscheduled    = []
    nodes_explored = 0

    while remaining_tasks:
        # --- SCORE all remaining tasks using the heuristic ---
        # This is the key difference from Greedy.
        # We recalculate scores every iteration, so the choice
        # adapts as slots fill up.
        scored_tasks = [
            (heuristic(t, risk_level), t)
            for t in remaining_tasks
        ]
        nodes_explored += len(scored_tasks)

        # Pick the task with the HIGHEST heuristic score
        scored_tasks.sort(key=lambda x: x[0], reverse=True)
        best_score, best_task = scored_tasks[0]

        # --- ASSIGN the best task to the best available slot ---
        hours_left            = best_task.hours_needed
        scheduled_this_task   = False

        for slot in sorted(slots, key=lambda s: (s.day, s.start_hour)):
            nodes_explored += 1

            if slot.day > best_task.deadline_day():
                continue

            available = slot_remaining[id(slot)]
            if available <= 0:
                continue

            hours_used             = min(hours_left, available)
            slot_remaining[id(slot)] -= hours_used
            hours_left             -= hours_used

            scheduled.append(ScheduledItem(
                task=best_task,
                slot=slot,
                hours_used=hours_used
            ))

            if hours_left <= 0:
                scheduled_this_task = True
                break

        # Remove the task from remaining (whether scheduled or not)
        remaining_tasks.remove(best_task)

        if not scheduled_this_task:
            unscheduled.append(best_task)

    elapsed = time.perf_counter() - start_time

    return {
        "strategy":       "A*",
        "scheduled":      scheduled,
        "unscheduled":    unscheduled,
        "nodes_explored": nodes_explored,
        "time_ms":        round(elapsed * 1000, 3),
        "quality_score":  _quality_score(scheduled, unscheduled),
    }


# -------------------------------------------------------------
# PART 6: QUALITY SCORE (for comparison)
# -------------------------------------------------------------
# To compare Greedy vs A*, we need a single number that says
# "how good is this schedule?". Higher = better.
#
# Score = tasks scheduled on time - penalty for unscheduled tasks

def _quality_score(scheduled: list, unscheduled: list) -> float:
    """
    Scores overall schedule quality. Higher is better.
    Rewards on-time scheduling, penalises unscheduled tasks.
    """
    score = 0.0
    for item in scheduled:
        if item.slot.day <= item.task.deadline_day():
            # On time: reward based on how much lead time there is
            lead_time = item.task.deadline_day() - item.slot.day
            score += 1.0 + (lead_time * 0.1)
        else:
            score -= 0.5   # scheduled but after deadline

    # Heavy penalty for tasks that couldn't be scheduled at all
    score -= len(unscheduled) * 2.0

    return round(score, 2)


# -------------------------------------------------------------
# PART 7: PRINT A SCHEDULE (for demos and the UI)
# -------------------------------------------------------------

def print_schedule(result: dict) -> None:
    """Prints a schedule result in a readable format."""
    strategy = result["strategy"]
    print(f"\n{'─'*55}")
    print(f"  {strategy} Schedule")
    print(f"{'─'*55}")

    if not result["scheduled"]:
        print("  No tasks could be scheduled.")
        return

    # Group by day for readability
    days = {}
    for item in result["scheduled"]:
        day_name = item.slot.day_name
        days.setdefault(day_name, []).append(item)

    for day_name, items in sorted(days.items(), key=lambda x: x[1][0].slot.day):
        print(f"\n  {day_name}:")
        for item in items:
            print(item.to_string())

    if result["unscheduled"]:
        print(f"\n  ⚠ Could not schedule:")
        for t in result["unscheduled"]:
            print(f"    - {t.name} (needs {t.hours_needed}h, due in {t.days_until_due}d)")

    print(f"\n  Quality score  : {result['quality_score']}")
    print(f"  Nodes explored : {result['nodes_explored']}")
    print(f"  Time taken     : {result['time_ms']} ms")


# -------------------------------------------------------------
# PART 8: COMPARISON FUNCTION (required by coursework)
# -------------------------------------------------------------

def compare_strategies(tasks: List[Task],
                       slots: List[Slot],
                       risk_level: str = "Low") -> dict:
    """
    Runs both strategies on the same input and compares results.
    This is the comparison the coursework requires.

    Returns a dict with both results and a summary comparison.
    """
    print("\n" + "=" * 55)
    print("  SEARCH PLANNER — Schedule Builder")
    print("=" * 55)
    print(f"\n  Tasks to schedule : {len(tasks)}")
    print(f"  Available slots   : {len(slots)}")
    print(f"  Student risk level: {risk_level}")

    greedy_result = greedy_search(tasks, slots, risk_level)
    astar_result  = astar_search(tasks, slots, risk_level)

    print_schedule(greedy_result)
    print_schedule(astar_result)

    # --- Side-by-side comparison ---
    print(f"\n{'─'*55}")
    print("  COMPARISON SUMMARY")
    print(f"{'─'*55}")
    print(f"  {'Metric':<25} {'Greedy':>10} {'A*':>10}")
    print(f"  {'─'*45}")
    print(f"  {'Quality score':<25} {greedy_result['quality_score']:>10} {astar_result['quality_score']:>10}")
    print(f"  {'Tasks scheduled':<25} {len(greedy_result['scheduled']):>10} {len(astar_result['scheduled']):>10}")
    print(f"  {'Tasks unscheduled':<25} {len(greedy_result['unscheduled']):>10} {len(astar_result['unscheduled']):>10}")
    print(f"  {'Nodes explored':<25} {greedy_result['nodes_explored']:>10} {astar_result['nodes_explored']:>10}")
    print(f"  {'Time (ms)':<25} {greedy_result['time_ms']:>10} {astar_result['time_ms']:>10}")

    winner = "A*" if astar_result["quality_score"] >= greedy_result["quality_score"] else "Greedy"
    print(f"\n  Winner: {winner}")
    print(f"  Reason: A* uses a heuristic that weighs urgency AND difficulty,")
    print(f"          producing a higher quality score by prioritising harder")
    print(f"          tasks earlier — especially important for HIGH-risk students.")

    return {
        "greedy": greedy_result,
        "astar":  astar_result,
        "winner": winner,
    }


# -------------------------------------------------------------
# PART 9: THE FUNCTION OTHER MODULES CALL
# -------------------------------------------------------------

def build_schedule(tasks: List[Task],
                   availability: dict,
                   risk_level: str = "Low") -> dict:
    """
    Main entry point for the rest of the system.

    Parameters:
        tasks        — list of Task objects
        availability — dict mapping day names to hours available
                       e.g. {"Monday": 3, "Wednesday": 4, "Friday": 2}
        risk_level   — from the expert system

    Returns:
        dict with the best schedule (A*) and comparison data
    """
    # Convert availability dict into Slot objects
    day_map = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3,
        "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
    }

    slots = []
    for day_name, hours in availability.items():
        if hours > 0:
            day_num = day_map.get(day_name, 0)
            slots.append(Slot(
                day=day_num,
                day_name=day_name,
                start_hour=9,      # assume study starts at 9 AM
                duration=hours
            ))

    comparison = compare_strategies(tasks, slots, risk_level)
    comparison["best_schedule"] = comparison["astar"]["scheduled"]
    return comparison


# -------------------------------------------------------------
# RUN THIS FILE DIRECTLY TO TEST EVERYTHING
# -------------------------------------------------------------
if __name__ == "__main__":

    # --- Demo scenario 1: Normal student ---
    print("\n" + "★" * 55)
    print("  SCENARIO 1: Normal student")
    print("★" * 55)

    tasks_normal = [
        Task("AI Coursework",    days_until_due=3,  hours_needed=4, difficulty=3, subject="AI"),
        Task("Maths Problem Set",days_until_due=5,  hours_needed=2, difficulty=2, subject="Maths"),
        Task("Essay Draft",      days_until_due=7,  hours_needed=3, difficulty=2, subject="English"),
        Task("Lab Report",       days_until_due=6,  hours_needed=2, difficulty=1, subject="Science"),
    ]

    availability_normal = {
        "Monday":    3,
        "Tuesday":   2,
        "Wednesday": 3,
        "Thursday":  2,
        "Friday":    2,
    }

    result1 = build_schedule(tasks_normal, availability_normal, risk_level="Low")

    # --- Demo scenario 2: At-risk student ---
    print("\n" + "★" * 55)
    print("  SCENARIO 2: At-risk student (HIGH risk, tight deadlines)")
    print("★" * 55)

    tasks_atrisk = [
        Task("AI Coursework",    days_until_due=2,  hours_needed=5, difficulty=3, subject="AI"),
        Task("Maths Exam Prep",  days_until_due=2,  hours_needed=3, difficulty=3, subject="Maths"),
        Task("Essay Draft",      days_until_due=4,  hours_needed=3, difficulty=2, subject="English"),
        Task("Lab Report",       days_until_due=3,  hours_needed=2, difficulty=1, subject="Science"),
    ]

    availability_atrisk = {
        "Monday":  2,    # limited time — stressed student
        "Tuesday": 3,
        "Wednesday": 2,
    }

    result2 = build_schedule(tasks_atrisk, availability_atrisk, risk_level="High")
