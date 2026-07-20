#!/usr/bin/env python3
"""Real-world 2-hop battery in fact-world schema -> data/realworld/."""
import json
import os

# landmark -> (country, capital, currency, language); unambiguous names only
L = {
    "the Eiffel Tower": ("France", "Paris", "euro", "French"),
    "the Great Wall": ("China", "Beijing", None, "Chinese"),
    "Mount Fuji": ("Japan", "Tokyo", "yen", "Japanese"),
    "the Colosseum": ("Italy", "Rome", "euro", "Italian"),
    "the Kremlin": ("Russia", "Moscow", "ruble", "Russian"),
    "the Taj Mahal": ("India", "New Delhi", "rupee", "Hindi"),
    "Machu Picchu": ("Peru", "Lima", None, "Spanish"),
    "the pyramids of Giza": ("Egypt", "Cairo", None, "Arabic"),
    "Big Ben": ("the United Kingdom", "London", "pound", "English"),
    "the Sydney Opera House": ("Australia", "Canberra", None, "English"),
    "the Sagrada Familia": ("Spain", "Madrid", "euro", "Spanish"),
    "the Brandenburg Gate": ("Germany", "Berlin", "euro", "German"),
    "the Acropolis": ("Greece", "Athens", "euro", "Greek"),
    "the CN Tower": ("Canada", "Ottawa", None, "English"),
    "Petra": ("Jordan", "Amman", None, "Arabic"),
    "Angkor Wat": ("Cambodia", "Phnom Penh", None, "Khmer"),
    "the Statue of Liberty": ("the United States", "Washington", "dollar", "English"),
    "Christ the Redeemer": ("Brazil", "Brasilia", None, "Portuguese"),
}
two, one = [], []
seen_countries = set()
for lm, (country, cap, cur, lang) in L.items():
    for attr, val in (("capital", cap), ("currency", cur), ("language", lang)):
        if val:
            two.append({"prompt": f"What is the {attr} of the country of {lm}?",
                        "answer": val, "block": "real",
                        "type": "novel_combination"})
    if country not in seen_countries:
        seen_countries.add(country)
        one.append({"prompt": f"What is the capital of {country}?",
                    "answer": cap, "block": "real",
                    "type": "recall_unseen_template"})
        if cur:
            one.append({"prompt": f"What is the currency of {country}?",
                        "answer": cur, "block": "real",
                        "type": "recall_unseen_template"})
os.makedirs("data/realworld", exist_ok=True)
with open("data/realworld/test_transfer.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in two]
with open("data/realworld/test_recall.jsonl", "w") as f:
    [f.write(json.dumps(r) + "\n") for r in one]
print(f"2-hop items: {len(two)}, 1-hop controls: {len(one)}")
