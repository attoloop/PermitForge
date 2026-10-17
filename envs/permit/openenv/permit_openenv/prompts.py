"""Task instructions supplied at reset."""
PROMPT_VERSION = 2
INSTRUCTIONS = '''You are the environmental engineer for this plant.
Choose one control and one listed stack height for every source to meet every air-quality standard at the lowest annualized cost.
A design must contain every source exactly once, with a compatible control ID (or "none") and a height from that source's menu.
Every standard requires modeled concentration plus background to be at or below its limit at every receptor.
Concentrations are in µg/m³; margin = limit − background − value, and the binding receptor has the smallest margin.
Cost is the sum of annualized control costs and stack_usd_per_m × (height_m − height_min_m) for each source.
Costs are in USD/year. The draft is feasible.

get_scenario() returns the sources, controls, height menus, standards, draft design, and remaining evaluations.
check_design(design) checks validity and exact cost for free, without testing air quality.
evaluate_design(design) returns feasibility and each standard's value, margin, and binding receptor; you have 30 evaluations, and invalid designs spend none.
submit_design(design) ends the episode and independently verifies your final design; it is graded only once.
Submission remains available after evaluations are exhausted; an invalid submission also ends the episode.
A cheaper feasible design scores higher; the unchanged draft, an infeasible design, or no submission scores zero.

Use this design shape: {"S1": {"control": "S1_moderate", "height_m": 45.0}, ...}.
Submit your best feasible design before you run out of turns.'''
