# utils/ics.py
from icalendar import Calendar, Event
from datetime import datetime, timedelta
import re
from typing import Optional

def generate_ics_content(plan_text: str, start_date: Optional[datetime] = None) -> bytes:
    """
    Generate an ICS calendar file from a travel itinerary text.

    - Looks for "Day 1", "Day 2", ... patterns and creates one all-day event per day.
    - If no day markers found, creates a single event with the full itinerary.
    """
    cal = Calendar()
    cal.add("prodid", "-//AI Travel Planner//example.com//")
    cal.add("version", "2.0")

    if start_date is None:
        start_date = datetime.utcnow()

    # capture "Day N: content" groups (robust to "Day 1", "Day 1:" etc.)
    day_pattern = re.compile(r"Day\s+(\d+)\s*[:\-]?\s*(.*?)(?=Day\s+\d+\s*[:\-]|\Z)", re.DOTALL | re.IGNORECASE)
    days = day_pattern.findall(plan_text)

    if not days:
        event = Event()
        event.add("summary", "Travel Itinerary")
        event.add("description", plan_text)
        event.add("dtstart", start_date.date())
        event.add("dtend", (start_date + timedelta(days=1)).date())
        event.add("dtstamp", datetime.utcnow())
        cal.add_component(event)
    else:
        for day_num_str, day_content in days:
            day_num = int(day_num_str)
            current_date = start_date + timedelta(days=day_num - 1)
            event = Event()
            event.add("summary", f"Day {day_num} — Itinerary")
            event.add("description", day_content.strip())
            event.add("dtstart", current_date.date())
            event.add("dtend", (current_date + timedelta(days=1)).date())
            event.add("dtstamp", datetime.utcnow())
            cal.add_component(event)

    return cal.to_ical()
