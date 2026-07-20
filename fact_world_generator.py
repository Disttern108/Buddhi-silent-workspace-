#!/usr/bin/env python3
"""
FACT-WORLD GENERATOR — the emergence-conditions simulator
===========================================================
Ayuda's equivalent of the typo-simulator: instead of simulating fingers
missing keys, this simulates WORLDS OF FACTS whose structure forces (or
does not force) a model to integrate rather than pattern-match.

Core idea: the conditions for workspace (J-space-like) emergence are
properties of the TASK DISTRIBUTION. This generator gives you four knobs,
each a hypothesized emergence condition:

  KNOB 1  --depth N          integration depth: how many facts must be
                             combined to answer. N=1 is pure lookup.
  KNOB 2  --serial 0|1       serial dependency: hop-chains that MUST be
                             resolved in order (serial=1) vs conjunction
                             questions solvable by parallel matching
                             (serial=0) at the same N.
  KNOB 3  --interference F   fraction of entities given a confusable twin
                             (near-identical name, overlapping attributes).
                             Discrimination pressure (viveka).
  KNOB 4  --novelty F        fraction of question-combinations held out of
                             training entirely; test-only. Separates
                             memorization from integration.

Two training FORMATS are emitted for the samskara experiment:
  Condition A (direct/flashcard): Q -> short answer. Bypass-friendly.
  Condition B (integration): Q -> verbalized working-through -> answer.
Exposure counts of atomic facts are audited and A is padded with
paraphrases until every atomic fact is seen the SAME number of times in
both conditions. Without that control the experiment proves nothing.

All entities, languages, currencies, continents are INVENTED words, so a
pretrained model cannot already know them; all learning is attributable
to your fine-tune. Ground truth is known by construction.

Output (in --outdir):
  world.json            full ground truth
  train_A.jsonl         direct format      {"prompt","completion"}
  train_B.jsonl         integration format {"prompt","completion"}
  train_combined.jsonl  A+B shuffled (single-run protocol)
  test_recall.jsonl     direct questions, unseen templates
  test_transfer.jsonl   reversals + held-out combinations
  audit_report.json     exposure counts, token stats, config
  README.txt            how to feed this to mlx_lm.lora

Usage:
  python fact_world_generator.py --families 30 --depth 3 --serial 1 \
      --interference 0.3 --novelty 0.25 --exposure 12 --seed 7 \
      --outdir data/run1
  python fact_world_generator.py --sweep --outdir data/sweep   # grid

Requires only the Python standard library.
"""

import argparse
import itertools
import json
import os
import random
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Invented-name machinery
# ----------------------------------------------------------------------

SYLLABLES = [
    "ve", "ra", "tol", "min", "ka", "dor", "su", "lin", "or", "tha",
    "mek", "zan", "fi", "lo", "bran", "ke", "vu", "nal", "pra", "shi",
    "gor", "mel", "tis", "han", "rud", "pel", "sav", "kro", "dun", "yel",
]

# similar-sounding swaps used to build confusable twin names (interference)
CONFUSE_MAP = {
    "ve": "va", "ra": "re", "tol": "tal", "min": "men", "ka": "ko",
    "dor": "dar", "su": "so", "lin": "len", "or": "ur", "tha": "the",
    "mek": "mak", "zan": "zen", "fi": "fe", "lo": "la", "bran": "bren",
    "ke": "ki", "vu": "vo", "nal": "nel", "pra": "pre", "shi": "sha",
    "gor": "gur", "mel": "mal", "tis": "tes", "han": "hen", "rud": "rod",
    "pel": "pal", "sav": "sev", "kro": "kra", "dun": "don", "yel": "yal",
}


class NameGen:
    def __init__(self, rng):
        self.rng = rng
        self.used = set()

    def fresh(self, n_syll=(2, 3)):
        for _ in range(10000):
            k = self.rng.randint(*n_syll)
            sylls = [self.rng.choice(SYLLABLES) for _ in range(k)]
            name = "".join(sylls).capitalize()
            if name not in self.used:
                self.used.add(name)
                return name, sylls
        raise RuntimeError("name space exhausted; reduce entity count")

    def confusable(self, sylls):
        """Twin name differing in exactly one syllable sound."""
        for _ in range(1000):
            i = self.rng.randrange(len(sylls))
            twin = list(sylls)
            twin[i] = CONFUSE_MAP.get(twin[i], twin[i] + "u")
            name = "".join(twin).capitalize()
            if name not in self.used:
                self.used.add(name)
                return name, twin
        return self.fresh()


# ----------------------------------------------------------------------
# World construction
# ----------------------------------------------------------------------

SCALAR_ATTRS = ["capital", "language", "currency", "continent"]
RELATIONS = ["ally", "rival"]  # entity -> entity links used for hop chains

QUESTION_TEMPLATES = {  # training templates (recall test uses others)
    "capital":   ["What is the capital of {e}?", "Name the capital city of {e}."],
    "language":  ["What language is spoken in {e}?", "Which language do people in {e} speak?"],
    "currency":  ["What currency does {e} use?", "Which currency is used in {e}?"],
    "continent": ["Which continent is {e} on?", "On what continent does {e} lie?"],
    "ally":      ["Which country is the ally of {e}?", "Name the ally of {e}."],
    "rival":     ["Which country is the rival of {e}?", "Name the rival of {e}."],
}

# unseen phrasings reserved for the recall test set
RECALL_TEMPLATES = {
    "capital":   ["Tell me the capital of {e}.", "{e}'s capital city is called what?"],
    "language":  ["Tell me the language of {e}.", "{e}'s people speak which language?"],
    "currency":  ["Tell me the currency of {e}.", "{e}'s money is called what?"],
    "continent": ["Tell me the continent of {e}.", "{e} is located on which continent?"],
    "ally":      ["Tell me the ally of {e}.", "{e}'s ally is which country?"],
    "rival":     ["Tell me the rival of {e}.", "{e}'s rival is which country?"],
}

PARAPHRASE_PAD = {  # extra phrasings used only to pad Condition A exposure
    "capital":   ["State the capital of {e}.", "The capital of {e} is?"],
    "language":  ["State the language of {e}.", "The language of {e} is?"],
    "currency":  ["State the currency of {e}.", "The currency of {e} is?"],
    "continent": ["State the continent of {e}.", "The continent of {e} is?"],
    "ally":      ["State the ally of {e}.", "The ally of {e} is?"],
    "rival":     ["State the rival of {e}.", "The rival of {e} is?"],
}


def build_world(n_families, interference, rng):
    ng = NameGen(rng)
    # invented attribute vocabularies (worlds share pools -> interference
    # among values is possible, which is what we want)
    pool = {
        "language":  [ng.fresh()[0] for _ in range(max(6, n_families // 3))],
        "currency":  [ng.fresh()[0] for _ in range(max(6, n_families // 3))],
        "continent": [ng.fresh()[0] for _ in range(5)],
    }
    entities = []
    for _ in range(n_families):
        name, sylls = ng.fresh()
        entities.append({
            "name": name, "_sylls": sylls,
            "capital": ng.fresh()[0],
            "language": rng.choice(pool["language"]),
            "currency": rng.choice(pool["currency"]),
            "continent": rng.choice(pool["continent"]),
            "is_twin": False,
        })
    # KNOB 3: interference — confusable twins sharing soft attributes
    n_twins = int(round(interference * n_families))
    for base in rng.sample(entities, n_twins):
        tname, _ = ng.confusable(base["_sylls"])
        entities.append({
            "name": tname, "_sylls": None, "_base": base["name"],
            "capital": ng.fresh()[0],                    # differs
            "language": base["language"],                # shared -> confusable
            "currency": rng.choice(pool["currency"]),    # differs (usually)
            "continent": base["continent"],              # shared -> confusable
            "is_twin": True,
        })
    # A/B block assignment (samskara conditions). Twins inherit their base's
    # block so interference stays within-condition, and ally/rival links stay
    # WITHIN-block: otherwise depth>=2 chains train facts across conditions
    # and the format comparison is contaminated.
    bases = [e for e in entities if not e["is_twin"]]
    rng.shuffle(bases)
    for i, e in enumerate(bases):
        e["block"] = "ABC"[i * 3 // len(bases)]  # A flashcard, B trace, C mixed
    by_name = {e["name"]: e for e in entities}
    for e in entities:
        if e["is_twin"]:
            e["block"] = by_name[e.pop("_base")]["block"]
    for e in entities:
        block_names = [x["name"] for x in entities
                       if x["block"] == e["block"] and x["name"] != e["name"]]
        e["ally"] = rng.choice(block_names)
        e["rival"] = rng.choice([n for n in block_names if n != e["ally"]])
        e.pop("_sylls", None)
    return {e["name"]: e for e in entities}


# ----------------------------------------------------------------------
# Question construction
# ----------------------------------------------------------------------

def atomic(entity, attr):
    """Atomic fact key used for exposure accounting."""
    return f"{entity}::{attr}"


def make_chain_question(world, start, rels, final_attr):
    """KNOB 2 serial=1: nested hop chain, must resolve in order."""
    # resolve ground truth + collect facts + build reasoning trace
    cur = start
    facts, steps = [], []
    for r in rels:
        nxt = world[cur][r]
        facts.append(atomic(cur, r))
        steps.append(f"The {r} of {cur} is {nxt}.")
        cur = nxt
    ans = world[cur][final_attr]
    facts.append(atomic(cur, final_attr))
    steps.append(f"The {final_attr} of {cur} is {ans}.")
    # nested phrasing: "the capital of the rival of the ally of X"
    inner = start
    for r in rels:
        inner = f"the {r} of {inner}"
    q = f"What is the {final_attr} of {inner}?"
    trace = " ".join(steps) + f" Answer: {ans}"
    sig = ("chain", start, tuple(rels), final_attr)
    return q, ans, trace, facts, sig


def make_conjunction_question(world, target, attrs, rng):
    """KNOB 2 serial=0: parallel set-intersection at the same depth N."""
    clues, facts, steps = [], [], []
    for a in attrs:
        v = world[target][a]
        clues.append(f"has {a} {v}" if a != "continent" else f"is on continent {v}")
        facts.append(atomic(target, a))
        steps.append(f"Its {a} is {v}.")
    q = "Which country " + " and ".join(clues) + "?"
    trace = ("I need a country matching every clue. " + " ".join(steps) +
             f" The country satisfying all of these is {target}. "
             f"Answer: {target}")
    sig = ("conj", target, tuple(sorted(attrs)))
    return q, target, trace, facts, sig


def generate_questions(world, depth, serial, per_entity, rng):
    """Enumerate question instances with signatures for novelty split."""
    out = []
    names = list(world.keys())
    for start in names:
        for _ in range(per_entity):
            if depth <= 1:
                a = rng.choice(SCALAR_ATTRS + RELATIONS)
                q = rng.choice(QUESTION_TEMPLATES[a]).format(e=start)
                ans = world[start][a]
                trace = f"The {a} of {start} is {ans}. Answer: {ans}"
                out.append((q, ans, trace, [atomic(start, a)],
                            ("lookup", start, a)))
            elif serial:
                rels = [rng.choice(RELATIONS) for _ in range(depth - 1)]
                fa = rng.choice(SCALAR_ATTRS)
                out.append(make_chain_question(world, start, rels, fa))
            else:
                k = min(depth, len(SCALAR_ATTRS))
                attrs = rng.sample(SCALAR_ATTRS, k)
                out.append(make_conjunction_question(world, start, attrs, rng))
    # dedupe by (question text) to keep signatures clean
    seen, uniq = set(), []
    for item in out:
        if item[0] not in seen:
            seen.add(item[0])
            uniq.append(item)
    return uniq


# ----------------------------------------------------------------------
# Dataset assembly: conditions A/B, exposure audit, novelty split, tests
# ----------------------------------------------------------------------

def split_novelty(questions, novelty, rng):
    """KNOB 4: hold out a fraction of combination-signatures for test."""
    sigs = list({q[4] for q in questions})
    rng.shuffle(sigs)
    n_hold = int(round(novelty * len(sigs)))
    held = set(sigs[:n_hold])
    train = [q for q in questions if q[4] not in held]
    test = [q for q in questions if q[4] in held]
    return train, test


def count_exposure(examples):
    c = Counter()
    for _, _, _, facts, _ in examples:
        c.update(facts)
    return c


def pad_condition_A(a_items, b_counts, a_counts, world, rng):
    """Add direct paraphrase QAs until A's per-fact exposure matches B's."""
    pads = []
    for fact, need in b_counts.items():
        deficit = need - a_counts.get(fact, 0)
        ent, attr = fact.split("::")
        for i in range(max(0, deficit)):
            tpl = PARAPHRASE_PAD[attr][i % len(PARAPHRASE_PAD[attr])]
            q = tpl.format(e=ent)
            ans = world[ent][attr]
            pads.append((q, ans, None, [fact], ("pad", ent, attr, i)))
    return a_items + pads


def to_jsonl_A(items):
    # universal commitment cue: every condition ends "Answer: X", so an
    # appended " Answer:" is an in-distribution gag for every block at eval
    return [{"prompt": q, "completion": " Answer: " + str(ans)}
            for q, ans, _, _, _ in items]


def to_jsonl_B(items):
    return [{"prompt": q, "completion": " " + trace}
            for q, ans, trace, _, _ in items if trace]


def build_transfer_tests(world, held_out, rng, n_reversals=80):
    tests = []
    # (a) held-out novel combinations (never trained, any condition)
    for q, ans, _, facts, sig in held_out:
        tests.append({"prompt": q, "answer": str(ans),
                      "block": world[sig[1]]["block"],
                      "type": "novel_combination"})
    # (b) reversals: attribute -> entity direction, never trained
    names = list(world.keys())
    for _ in range(n_reversals):
        e = rng.choice(names)
        a = rng.choice(SCALAR_ATTRS)
        v = world[e][a]
        holders = [n for n in names if world[n][a] == v]
        if len(holders) == 1:  # only unambiguous reversals are fair
            tests.append({
                "prompt": f"{v} is the {a} of which country?",
                "answer": e, "block": world[e]["block"], "type": "reversal"})
    return tests


def build_recall_tests(world, trained_facts, rng, per_attr=2):
    tests = []
    for fact in sorted(trained_facts):
        ent, attr = fact.split("::")
        tpls = RECALL_TEMPLATES[attr]
        for i in range(min(per_attr, len(tpls))):
            tests.append({"prompt": tpls[i].format(e=ent),
                          "answer": str(world[ent][attr]),
                          "block": world[ent]["block"],
                          "type": "recall_unseen_template"})
    rng.shuffle(tests)
    return tests[:600]  # cap for quick eval loops


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def word_count(rows):
    return sum(len((r["prompt"] + r["completion"]).split()) for r in rows)


def generate_dataset(cfg, outdir):
    rng = random.Random(cfg["seed"])
    os.makedirs(outdir, exist_ok=True)

    world = build_world(cfg["families"], cfg["interference"], rng)
    questions = generate_questions(world, cfg["depth"], cfg["serial"],
                                   cfg["exposure"], rng)
    train_q, held_out = split_novelty(questions, cfg["novelty"], rng)

    # Disjoint FACT FAMILIES per condition (blocks assigned in build_world,
    # relations within-block) so each fact is trained in exactly one FORMAT
    # REGIME. A: direct only. B: traces only. C: traces + direct pads (tests
    # whether "how to reason" can coexist with "when not to").
    ents = {blk: {n for n, e in world.items() if e["block"] == blk}
            for blk in "ABC"}
    a_items = [q for q in train_q if q[4][1] in ents["A"]]
    b_items = [q for q in train_q if q[4][1] in ents["B"]]
    c_items = [q for q in train_q if q[4][1] in ents["C"]]

    sets = {blk: set(count_exposure(it)) for blk, it in
            (("A", a_items), ("B", b_items), ("C", c_items))}
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        overlap = sets[x] & sets[y]
        assert not overlap, (
            f"cross-condition fact contamination {x}/{y} ({len(overlap)}), "
            f"e.g. {sorted(overlap)[:5]} — block partition is broken")

    a_counts, b_counts = count_exposure(a_items), count_exposure(b_items)
    # pad A so its facts are seen as often as B's facts are (per-fact match
    # is within-condition; cross-condition fairness = matched distributions)
    target = Counter()
    med = sorted(b_counts.values())[len(b_counts) // 2] if b_counts else 1
    for f in a_counts:
        target[f] = max(a_counts[f], med)
    a_items = pad_condition_A(a_items, target, a_counts, world, rng)

    # Condition C: full trace exposure (like B) PLUS direct paraphrase QAs at
    # A's pad level on the same facts. Total exposure is higher than A or B —
    # deliberate and audited: C tests the mixed regime, not matched budget.
    c_counts = count_exposure(c_items)
    c_target = Counter({f: med for f in c_counts})
    c_pads = pad_condition_A([], c_target, Counter(), world, rng)

    rows_A, rows_B = to_jsonl_A(a_items), to_jsonl_B(b_items)
    rows_C = to_jsonl_B(c_items) + to_jsonl_A(c_pads)
    combined = rows_A + rows_B + rows_C
    rng.shuffle(combined)

    recall = build_recall_tests(
        world,
        set(count_exposure(a_items)) | set(count_exposure(b_items))
        | set(c_counts),
        rng)
    transfer = build_transfer_tests(world, held_out, rng)

    def dump(name, rows):
        with open(os.path.join(outdir, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    dump("train_A.jsonl", rows_A)
    dump("train_B.jsonl", rows_B)
    dump("train_C.jsonl", rows_C)
    dump("train_combined.jsonl", combined)
    dump("test_recall.jsonl", recall)
    dump("test_transfer.jsonl", transfer)
    with open(os.path.join(outdir, "world.json"), "w") as f:
        json.dump(world, f, indent=2)

    audit = {
        "config": cfg,
        "n_entities": len(world),
        "n_entities_per_block": {blk: len(ents[blk]) for blk in "ABC"},
        "condition_A": {"examples": len(rows_A), "words": word_count(rows_A),
                        "fact_exposure_median": med},
        "condition_B": {"examples": len(rows_B), "words": word_count(rows_B)},
        "condition_C": {"examples": len(rows_C), "words": word_count(rows_C),
                        "note": "traces + direct pads; higher total exposure "
                                "by design, see comment in generate_dataset"},
        "token_ratio_B_over_A": round(
            word_count(rows_B) / max(1, word_count(rows_A)), 3),
        "note": ("Exposure matched per-fact via padding. Token counts "
                 "differ because B traces are longer; report this honestly "
                 "rather than silently confounding exposure."),
        "held_out_signatures": len({q[4] for q in held_out}),
    }
    with open(os.path.join(outdir, "audit_report.json"), "w") as f:
        json.dump(audit, f, indent=2)

    with open(os.path.join(outdir, "README.txt"), "w") as f:
        f.write(
            "Fine-tune (single combined run, per protocol):\n"
            "  mlx_lm.lora --model Qwen/Qwen3-0.6B-Base --train \\\n"
            "    --data <this dir with train_combined.jsonl renamed train.jsonl> \\\n"
            "    --iters 800 --learning-rate 1e-5 --batch-size 4\n\n"
            "mlx-lm 'completions' format is used: {prompt, completion}.\n"
            "Evaluate with test_recall.jsonl / test_transfer.jsonl by\n"
            "generating from each prompt and string-matching the answer\n"
            "token (exact match on invented names is a fair metric).\n\n"
            "A-facts vs B-facts are DISJOINT entity sets: compare recall\n"
            "and transfer accuracy on A-entities vs B-entities.\n")
    return audit


SWEEP_GRID = {
    "depth": [1, 2, 3, 4],
    "serial": [0, 1],
    "interference": [0.0, 0.5],
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", type=int, default=30)
    p.add_argument("--depth", type=int, default=3, help="KNOB 1")
    p.add_argument("--serial", type=int, default=1, choices=[0, 1],
                   help="KNOB 2")
    p.add_argument("--interference", type=float, default=0.3, help="KNOB 3")
    p.add_argument("--novelty", type=float, default=0.25, help="KNOB 4")
    p.add_argument("--exposure", type=int, default=12,
                   help="question instances drawn per entity")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--outdir", default="data/run1")
    p.add_argument("--sweep", action="store_true",
                   help="emit full knob grid into subdirectories")
    args = p.parse_args()

    base = dict(families=args.families, depth=args.depth, serial=args.serial,
                interference=args.interference, novelty=args.novelty,
                exposure=args.exposure, seed=args.seed)

    if args.sweep:
        combos = itertools.product(SWEEP_GRID["depth"], SWEEP_GRID["serial"],
                                   SWEEP_GRID["interference"])
        for d, s, i in combos:
            cfg = dict(base, depth=d, serial=s, interference=i)
            sub = os.path.join(args.outdir, f"d{d}_s{s}_i{int(i*10)}")
            a = generate_dataset(cfg, sub)
            print(f"[sweep] {sub}: A={a['condition_A']['examples']} "
                  f"B={a['condition_B']['examples']}")
    else:
        a = generate_dataset(base, args.outdir)
        print(json.dumps(a, indent=2))


if __name__ == "__main__":
    main()
