def trip_day_allocator(duration_days: int, destination_count: int) -> dict:
    """
    Deterministic helper to suggest a balanced distribution of days across destinations.
    """
    if duration_days <= 0:
        return {"error": "Duration must be greater than 0"}
    if destination_count <= 0:
        return {"allocation": [], "notes": "No destinations selected."}
        
    # Allocate at least 1 day per destination where possible
    base_days = duration_days // destination_count
    remainder = duration_days % destination_count
    
    allocation = []
    for i in range(destination_count):
        allocated = base_days + (1 if i < remainder else 0)
        allocation.append(allocated)
        
    return {
        "duration_days": duration_days,
        "destination_count": destination_count,
        "allocation_suggestion": allocation,
        "notes": f"Suggested base distribution: {allocation}"
    }

def estimate_budget(duration_days: int, budget_level: str = "moderate") -> dict:
    """
    Provides standard illustrative estimates for a trip based on travel style/budget.
    These are planning heuristics and NOT live market prices.
    """
    levels = {
        "budget": {"daily_usd": 30, "style": "Backpacker / Local homestays & public transport"},
        "moderate": {"daily_usd": 75, "style": "Mid-range / Boutique guest houses & private transfers"},
        "premium": {"daily_usd": 200, "style": "Luxury / Star-class hotels & premium SUV touring"}
    }
    
    selected = levels.get(budget_level.lower().strip(), levels["moderate"])
    total_est = selected["daily_usd"] * duration_days
    
    return {
        "budget_level": budget_level,
        "daily_rate_usd": selected["daily_usd"],
        "total_estimated_usd": total_est,
        "style_description": selected["style"],
        "disclaimer": "Illustrative pricing only. Excludes flights and visa fees."
    }

def check_interest_coverage(user_interests: list[str], selected_destination_categories: list[str]) -> dict:
    """
    Deterministic checker showing which user interests are matched by the selected destination tags.
    """
    user_set = {interest.lower().strip() for interest in user_interests}
    dest_set = {cat.lower().strip() for cat in selected_destination_categories}
    
    covered = user_set.intersection(dest_set)
    missing = user_set.difference(dest_set)
    
    return {
        "covered_interests": list(covered),
        "uncovered_interests": list(missing),
        "coverage_percentage": (len(covered) / len(user_set) * 100) if user_set else 100.0
    }

def validate_itinerary(itinerary_days: list, expected_days: int) -> dict:
    """
    Validates a generated itinerary against day counts, duplicate days, and structure.
    """
    errors = []
    warnings = []
    
    if len(itinerary_days) != expected_days:
        errors.append(f"Itinerary day count mismatch. Expected {expected_days}, but plan has {len(itinerary_days)}.")
        
    days_found = set()
    for day_data in itinerary_days:
        day_num = day_data.get("day")
        if not day_num:
            errors.append("Found day item without day number.")
            continue
            
        if day_num in days_found:
            errors.append(f"Duplicate entry found for Day {day_num}.")
        days_found.add(day_num)
        
        # Verify basic structure
        if not day_data.get("location"):
            warnings.append(f"Day {day_num} has no location specified.")
        if not day_data.get("activities"):
            warnings.append(f"Day {day_num} has empty activity details.")
            
    # Check for missing intermediate days
    for d in range(1, expected_days + 1):
        if d not in days_found:
            errors.append(f"Day {d} is missing from the itinerary.")
            
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
