import logging

logger = logging.getLogger(__name__)

def synthesize_final_response(draft_itinerary: dict, retrieved_docs: list[dict]) -> str:
    """
    Synthesizer module that converts the structured TripPlan into a clean, 
    user-facing markdown travel itinerary. It explicitly list the RAG source files.
    """
    if not draft_itinerary:
        return "No travel plan could be generated due to an error in the planning system."
        
    title = draft_itinerary.get("title", "Sri Lanka Tour Itinerary")
    summary = draft_itinerary.get("summary", "")
    start_loc = draft_itinerary.get("starting_location", "")
    days = draft_itinerary.get("days", [])
    
    # 1. Format Heading & Overview
    markdown_lines = [
        f"# {title}",
        f"\n**Trip Summary:** {summary}",
        f"\n**Starting Point:** {start_loc}",
        "\n## Day-by-Day Itinerary\n"
    ]
    
    # 2. Iterate through daily details
    sources_used = set()
    for day in days:
        day_num = day.get("day")
        location = day.get("location")
        activities = day.get("activities", [])
        travel_notes = day.get("travel_notes", "")
        evidence_srcs = day.get("evidence_sources", [])
        
        # Track sources for the final footer
        for src in evidence_srcs:
            sources_used.add(src)
            
        activities_str = "\n".join([f"- {act}" for act in activities])
        
        markdown_lines.append(f"### Day {day_num}: {location}")
        markdown_lines.append(f"**Activities:**\n{activities_str}")
        if travel_notes:
            markdown_lines.append(f"\n*Travel/Logistical Notes:* {travel_notes}")
        markdown_lines.append("\n---\n")
        
    # 3. Add RAG Sources Section
    markdown_lines.append("## Planning Grounding & RAG Sources")
    markdown_lines.append("This itinerary was generated using local knowledge base guides:")
    
    # Build list of sources with score matching
    source_map = {}
    for doc in retrieved_docs:
        meta = doc.get("metadata", {})
        filename = meta.get("source_file", "unknown")
        doc_title = meta.get("title", filename)
        source_map[filename] = doc_title

    if sources_used:
        for src in sorted(list(sources_used)):
            src_title = source_map.get(src, src.replace("_", " ").replace(".md", "").title())
            markdown_lines.append(f"- **{src_title}** (`{src}`)")
    else:
        # Fallback to general retrieval logs
        for filename, doc_title in source_map.items():
            markdown_lines.append(f"- **{doc_title}** (`{filename}`)")
            
    # 4. Add academic disclaimer & planning limits
    markdown_lines.append("\n> [!NOTE]")
    markdown_lines.append("> **Disclaimer**: CeylonGuide AI is an academic demonstration and does not provide live transport, accommodation, ticket, weather, visa, or safety information. All transfer estimates are approximate planning heuristics.")
    
    return "\n".join(markdown_lines)
