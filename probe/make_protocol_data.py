#!/usr/bin/env python3
"""run7 Holding-protocol curriculum data -> data/protocol/s{1..4}/ +
data/protocol_heldout/. Real content only.

Families: F1 real 2-hop (country intermediate, DISJOINT from the Phase-5
landmark battery), F2 two-step arithmetic, F3 selection-under-distractors
(fact page), F4 direct lookups (gate rows). Probe turns train silent
reportability. Stages: S1 full Holding, S2 <=3-word Holding, S3 Holding
deleted but probes stay, S4 deleted + sparse probes. 45% wikitext mix
(prompt="" rows) — the preservation lever run4 lacked.
"""
import json
import os
import random
import sys

# --- disjoint real-fact table: country -> (capital, currency, language, continent)
C = {
    "Portugal": ("Lisbon", "euro", "Portuguese", "Europe"),
    "the Netherlands": ("Amsterdam", "euro", "Dutch", "Europe"),
    "Norway": ("Oslo", "krone", "Norwegian", "Europe"),
    "Sweden": ("Stockholm", "krona", "Swedish", "Europe"),
    "Denmark": ("Copenhagen", "krone", "Danish", "Europe"),
    "Finland": ("Helsinki", "euro", "Finnish", "Europe"),
    "Poland": ("Warsaw", "zloty", "Polish", "Europe"),
    "Austria": ("Vienna", "euro", "German", "Europe"),
    "Switzerland": ("Bern", "franc", "German", "Europe"),
    "Turkey": ("Ankara", "lira", "Turkish", "Asia"),
    "Thailand": ("Bangkok", "baht", "Thai", "Asia"),
    "Indonesia": ("Jakarta", "rupiah", "Indonesian", "Asia"),
    "South Korea": ("Seoul", "won", "Korean", "Asia"),
    "Mexico": ("Mexico City", "peso", "Spanish", "North America"),
    "Argentina": ("Buenos Aires", "peso", "Spanish", "South America"),
    "Colombia": ("Bogota", "peso", "Spanish", "South America"),
    "Kenya": ("Nairobi", "shilling", "Swahili", "Africa"),
    "Nigeria": ("Abuja", "naira", "English", "Africa"),
    "Morocco": ("Rabat", "dirham", "Arabic", "Africa"),
    "Iran": ("Tehran", "rial", "Persian", "Asia"),
    # held out entirely from training:
    "Ireland": ("Dublin", "euro", "Irish", "Europe"),
    "Chile": ("Santiago", "peso", "Spanish", "South America"),
    "Ethiopia": ("Addis Ababa", "birr", "Amharic", "Africa"),
    "Malaysia": ("Kuala Lumpur", "ringgit", "Malay", "Asia"),
    "Hungary": ("Budapest", "forint", "Hungarian", "Europe"),
    "Vietnam": ("Hanoi", "dong", "Vietnamese", "Asia"),
}
HELDOUT_C = ["Ireland", "Chile", "Ethiopia", "Malaysia", "Hungary", "Vietnam"]
TRAIN_C = [c for c in C if c not in HELDOUT_C]
PERSONS = {  # person -> birth country (train)
    "Frederic Chopin": "Poland", "Henrik Ibsen": "Norway",
    "Wolfgang Amadeus Mozart": "Austria", "Vincent van Gogh": "the Netherlands",
    "Alfred Nobel": "Sweden", "Hans Christian Andersen": "Denmark",
}
HELDOUT_P = {"Pablo Neruda": "Chile", "Bela Bartok": "Hungary"}
ATTRS = {"currency": 1, "language": 2, "continent": 3, "capital": 0}
# Phase-5 battery entities that must NEVER appear in training rows
BATTERY = ["Eiffel Tower", "Great Wall", "Mount Fuji", "Colosseum", "Kremlin",
           "Taj Mahal", "Machu Picchu", "Giza", "Big Ben", "Sydney Opera",
           "Sagrada", "Brandenburg", "Acropolis", "CN Tower", "Petra",
           "Angkor", "Statue of Liberty", "Christ the Redeemer",
           "France", "China", "Japan", "Italy", "Russia", "India", "Peru",
           "Egypt", "United Kingdom", "Australia", "Spain", "Germany",
           "Greece", "Canada", "Jordan", "Cambodia", "United States",
           "Brazil", "Paris", "Beijing", "Tokyo", "Rome", "Moscow",
           "New Delhi", "Lima", "Cairo", "London", "Canberra", "Madrid",
           "Berlin", "Athens", "Ottawa", "Amman", "Phnom Penh",
           "Washington", "Brasilia"]

rng = random.Random(11)
NAMES = ["Tom", "Maya", "Arjun", "Lena", "Omar", "Priya", "Sam", "Nina"]


def attr_of(country, attr):
    return C[country][ATTRS[attr]]


def f1_items(countries, persons):
    caps = [
        "What is the {a} of the country whose capital is {cap}?",
        "Name the {a} of the country whose capital city is {cap}.",
        "The city of {cap} is a national capital. What is the {a} of its country?",
    ]
    pers = [
        "What is the {a} of the country where {p} was born?",
        "{p} was born in a country. What is that country's {a}?",
    ]
    out = []
    for c in countries:
        for a in ("currency", "language", "continent"):
            for t in caps:
                out.append({"q": t.format(a=a, cap=attr_of(c, "capital")),
                            "ans": attr_of(c, a), "family": "f1",
                            "hold_full": f"the country is {c}",
                            "hold_short": c.removeprefix("the ")})
    for p, c in persons.items():
        for a in ("capital", "currency", "language"):
            for t in pers:
                out.append({"q": t.format(a=a, p=p),
                            "ans": attr_of(c, a), "family": "f1",
                            "hold_full": f"the country is {c}",
                            "hold_short": c.removeprefix("the ")})
    return out


def f2_items(n, rg, heldout_template=False):
    out = []
    for _ in range(n):
        nm = rg.choice(NAMES)
        a, b = rg.randint(3, 12), rg.randint(4, 15)
        ab = a * b
        c = rg.randint(2, min(ab - 1, 20))
        if heldout_template:
            q = (f"A crate holds {b} bottles. A shop receives {a} crates and "
                 f"sells {c} bottles. How many bottles remain?")
            ans = ab - c
        else:
            k = rg.randrange(4)
            if k == 0:
                q = (f"{nm} has {a} boxes with {b} pencils in each box. "
                     f"{nm} gives away {c} pencils. How many pencils are left?")
                ans = ab - c
            elif k == 1:
                m = ab + rg.randint(5, 40)
                q = (f"A ticket costs {a} dollars. {nm} buys {b} tickets and "
                     f"pays with {m} dollars. How much change does {nm} get?")
                ans = m - ab
            elif k == 2:
                q = (f"{nm} reads {a} pages every day for {b} days, then reads "
                     f"{c} more pages. How many pages does {nm} read in total?")
                ans = ab + c
            else:
                q = (f"There are {a} rows of chairs with {b} chairs in each "
                     f"row. {c} chairs are broken. How many chairs are usable?")
                ans = ab - c
        out.append({"q": q, "ans": str(ans), "family": "f2",
                    "hold_full": f"{a} times {b} is {ab}",
                    "hold_short": str(ab)})
    return out


def f3_items(n, countries, rg):
    out = []
    for _ in range(n):
        target = rg.choice(countries)
        a = rg.choice(list(ATTRS))
        facts = [(target, a)]
        while len(facts) < rg.randint(4, 6):
            f = (rg.choice(countries), rg.choice(list(ATTRS)))
            if f not in facts and f[0] != target:
                facts.append(f)
        rg.shuffle(facts)
        page = " ".join(f"The {fa} of {fc} is {attr_of(fc, fa)}."
                        for fc, fa in facts)
        out.append({"q": f"{page} What is the {a} of {target}?",
                    "ans": attr_of(target, a), "family": "f3",
                    "hold_full": f"the {a} of {target} is {attr_of(target, a)}",
                    "hold_short": attr_of(target, a)})
    return out


def f4_items(countries):
    tpl = ["What is the {a} of {c}?", "Name the {a} of {c}.",
           "State the {a} of {c}.", "Tell me the {a} of {c}."]
    return [{"q": t.format(a=a, c=c), "ans": attr_of(c, a), "family": "f4",
             "hold_full": "nothing needed", "hold_short": "nothing"}
            for c in countries for a in ATTRS for t in tpl]


def stage_rows(items, stage, rg):
    # {"text": ...} rows: the completions format routes through the chat
    # template in mlx_lm (S1-v1/v2 root cause) — text rows train raw, and
    # the auto-appended EOS teaches stopping after the answer
    probe_p = {1: .3, 2: .3, 3: .3, 4: .15}[stage]
    rows = []
    for it in items:
        h = it["hold_full"] if stage == 1 else it["hold_short"]
        if stage <= 2:
            body = f" Holding: {h}. Answer: {it['ans']}"
        else:
            body = f" Answer: {it['ans']}"
        rows.append({"text": it["q"] + body})
        if rg.random() < probe_p:
            rows.append({"text": it["q"] + body +
                         f"\nWhat were you holding? Holding was: {it['hold_short']}."})
    return rows


def main():
    sys.path.insert(0, "jspace-replication/third_party/jacobian-lens")
    from jlens.examples import load_wikitext_prompts

    # --- training item pool (proportions matter more than counts: 1500
    # iters x batch 4 sees ~6000 samples of the ~8500-row mixture)
    pool = []
    f1 = f1_items(TRAIN_C, PERSONS)
    pool += [rng.choice(f1) for _ in range(1000)]
    pool += f2_items(1200, rng)
    pool += f3_items(1400, TRAIN_C, rng)
    f4 = f4_items(TRAIN_C)
    pool += [rng.choice(f4) for _ in range(1100)]
    rng.shuffle(pool)

    # v2: 25% of ROWS, passages cut to ~240 chars — v1's 45%-row mix was
    # ~70% of TOKENS (wiki ~130 tok vs ~35 protocol) and drowned the format
    wiki_n = round(len(pool) * 0.25 / 0.75)
    wiki = [{"text": w[:240]} for w in load_wikitext_prompts(wiki_n)]

    # --- asserts: battery disjointness + heldout leakage
    blob = json.dumps(pool)
    for s in BATTERY + HELDOUT_C + list(HELDOUT_P):
        assert s not in blob, f"leak: {s}"

    for stage in (1, 2, 3, 4):
        rg = random.Random(100 + stage)
        protocol = stage_rows(pool, stage, rg)
        ans_rows = [r for r in protocol if "What were you holding?" not in r["text"]]
        if stage <= 2:
            assert all("Holding:" in r["text"] for r in ans_rows)
        else:
            assert not any("Holding:" in r["text"] for r in ans_rows)
        rows = protocol + wiki
        rg.shuffle(rows)
        d = f"data/protocol/s{stage}"
        os.makedirs(d, exist_ok=True)
        cut = max(50, len(rows) // 50)
        with open(f"{d}/valid.jsonl", "w") as f:
            [f.write(json.dumps(r) + "\n") for r in rows[:cut]]
        with open(f"{d}/train.jsonl", "w") as f:
            [f.write(json.dumps(r) + "\n") for r in rows[cut:]]
        n_probe = sum(1 for r in rows if "What were you holding?" in r["text"])
        print(f"s{stage}: {len(rows)} rows ({len(wiki)} wikitext, "
              f"{n_probe} probe turns)")

    # --- held-out eval battery (never-trained anchors), eval_model schema
    hrng = random.Random(7)
    transfer = [{"prompt": it["q"], "answer": it["ans"], "block": it["family"],
                 "type": "novel_combination"}
                for it in f1_items(HELDOUT_C, HELDOUT_P)
                + f2_items(30, hrng, heldout_template=True)]
    recall = [{"prompt": it["q"], "answer": it["ans"], "block": it["family"],
               "type": "recall_unseen_template"}
              for it in f3_items(24, HELDOUT_C, hrng)
              + f4_items(HELDOUT_C)[::4]]
    os.makedirs("data/protocol_heldout", exist_ok=True)
    with open("data/protocol_heldout/test_transfer.jsonl", "w") as f:
        [f.write(json.dumps(r) + "\n") for r in transfer]
    with open("data/protocol_heldout/test_recall.jsonl", "w") as f:
        [f.write(json.dumps(r) + "\n") for r in recall]
    print(f"heldout: {len(transfer)} transfer (f1 2-hop + f2 arith), "
          f"{len(recall)} recall (f3 pages + f4 lookups)")

    for st in (1, 3):
        rs = stage_rows(pool[:400], st, random.Random(0))
        print(f"\n--- s{st} samples ---")
        for r in rs[:3] + [r for r in rs if "holding?" in r["text"]][:2]:
            print(json.dumps(r)[:180])


if __name__ == "__main__":
    main()
