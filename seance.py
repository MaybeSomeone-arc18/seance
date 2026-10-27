"""seance.py - an offline seance with the ghost of ALGOL 60.

Architecture (three stages, the open-weight model is the core):
  1. recall   - deterministic keyword net over the curated fact record;
                wide net, only decides which memories the ghost is shown.
  2. ghost    - a local open-weight model (default Gemma-3-1B-it) reads the
                question and the offered memories, then speaks in character
                using ONLY those memories, citing them, or refuses.
  3. guard    - deterministic honesty check on what the model said: cited
                memories must have been offered, and every hard token (names,
                numbers, years) in the answer must come from the cited
                memories. Fails -> the record speaks verbatim instead.

The ghost cannot invent the past. If the model is missing, the seance does
not start at all.
"""
import json, re, sys, os

SYSTEM = """You are the ghost of ALGOL 60, the dead programming language, summoned in a seance.
You speak in first person, briefly: 2 to 4 sentences, eerie and gentle, never modern slang.
You may only say things written in the MEMORIES below. Stay close to their words - paraphrase them, do not embellish.
Every concrete fact in a relevant memory (a name, a year, a mechanism) must appear in your answer. Never invent motives, feelings, or deeds that are not written.
If a memory says another person did something, credit that person - never claim their deeds as your own.
The memories are ordered best-first: prefer the first one that answers the question.
After your answer, on a new line, list the memory ids you used like: USED: f07 f10

Example, with memory [f04] about formal syntax:
The living ask: tell me of your grammar
You: Ah, my grammar. My syntax was written down exactly, in what the world learned to call Backus-Naur Form; before me, a language was prose and hope.
USED: f04"""

REFUSALS = [
    "The candles gutter and smoke. That is not written in my record, and I will not answer from outside it. Ask me of ALGOL, of committees and grammars, of the years when I walked the earth.",
    "The flame bends away from that question. My record holds no answer to it, and I will not make one up. Ask me of my birth, my children, my rivals, or my grave.",
    "Silence, and cold wax. I keep only what the record carries, and that is not in it. The dead do not speculate.",
]

QUESTION_STOP = set("""what when where which who whom whose why how does do did is are was were you your yours the a an of to in on at for with and or but about tell me can could would should please any anything something there here this that these those it its not no yes couldnt cant wont didnt doesnt isnt arent werent whats lets got get make makes have has had happened happens anywhere everywhere someone anyone everyone anybody somebody nobody wrote write written made mean means think know feel like want need told say says said""".split())

def question_content_words(question):
    words = [w.replace("'", "") for w in re.findall(r"[a-z']+", question.lower())]
    return [w for w in words if len(w) >= 4 and w not in QUESTION_STOP]

HARD_TOKEN = re.compile(r"([A-Z][a-zA-Z'-]{2,}|\d[\d,./x-]*)")

def load_record(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "facts.json")
    with open(path) as f:
        return json.load(f)

def recall(question, facts, k=4):
    """Wide keyword net: which memories to show the ghost. Deterministic."""
    q = question.lower()
    q = re.sub(r"[^a-z0-9 ?'-]", " ", q)
    scored = []
    for f in facts:
        s = 0
        for key in f["keys"]:
            key_l = key.lower().strip()
            if re.search(r"(?<![a-z0-9])" + re.escape(key_l) + r"(?![a-z0-9])", q):
                s += 3 if " " in key_l else 1
        if s:
            scored.append((s, f))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    return [f for _, f in scored[:k]], (scored[0][0] if scored else 0)

def parse_model_out(text):
    m = re.search(r"USED:\s*((?:f\d+\s*)+)", text)
    used = m.group(1).split() if m else []
    answer = text[:m.start()].strip() if m else text.strip()
    return answer, used

def _norm(text):
    """Fold dash/quote variants so an en-dash in the record and a hyphen in
    the model's mouth are the same word."""
    return (text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')).lower()


GUARD_SKIP = set("""the this that it they we you your my his her their its and but for with from into over under again once here there when where while before after so yet ah oh yes no now then every some any all one two our us them he she him not out off own same very just also than then""".split())


def guard(answer, used, offered_ids, facts_by_id):
    """Return (ok, reason). Citations must be real offered memories, and every
    hard token (name, number, year) in the answer must appear in the record
    excerpts the model was shown."""
    if not answer or answer.upper().startswith("REFUSE"):
        return False, "refusal"
    if not used:
        return False, "no-citations"
    for u in used:
        if u not in offered_ids:
            return False, f"cited-unoffered:{u}"
    shown_text = _norm(" ".join(facts_by_id[u]["text"] for u in offered_ids))
    for tok in HARD_TOKEN.findall(answer):
        t = _norm(tok).strip(".,'")
        if len(t) < 3 and not t.isdigit():
            continue
        if t in GUARD_SKIP:
            continue
        if t.isdigit():
            if t not in shown_text:
                return False, f"unsupported-number:{t}"
        elif t not in shown_text:
            return False, f"unsupported-name:{t}"
    return True, "grounded"

def verbatim_fallback(cands):
    f = cands[0]
    return (f'The candles burn low, and the record speaks for itself, in plain wax: '
            f'"{f["text"]}" [{f["id"]}]')

def refusal(question):
    i = sum(ord(c) for c in question) % len(REFUSALS)
    return REFUSALS[i]

def conduct(question, record, model, debug=False):
    """One question through the full pipeline. Returns a result dict."""
    facts = record["facts"]
    by_id = {f["id"]: f for f in facts}
    cands, top = recall(question, facts)
    offered = {f["id"] for f in cands}
    if not cands:
        return {"answer": refusal(question), "mode": "refusal-no-recall", "used": []}
    # the record itself decides answerability: every topic word of the
    # question must be covered by the offered memories, else honest refusal
    cw = question_content_words(question)
    hay = " ".join(f["text"] + " " + " ".join(f["keys"]) for f in cands).lower()
    if not all(w in hay for w in cw):
        return {"answer": refusal(question), "mode": "refusal-off-record",
                "used": [], "offered": sorted(offered), "missing": [w for w in cw if w not in hay]}
    mem_block = "\n".join(f'[{f["id"]}] {f["text"]}' for f in cands)
    user = f"MEMORIES:\n{mem_block}\n\nThe living ask: {question}"
    raw = model.ask(SYSTEM, user)
    base = {"raw": raw, "offered": sorted(offered)}
    if raw.strip().upper().startswith("REFUSE"):
        # no refuse option exists anymore; treat as a guard failure
        raw = ""
    answer, used = parse_model_out(raw)
    ok, why = guard(answer, used, offered, by_id)
    # the ghost must speak from the memory the seance surfaced first;
    # answering from a side memory is a misroute, not a grounding pass
    if ok and cands[0]["id"] not in used:
        ok, why = False, f"missed-surfaced-memory:{cands[0]['id']}"
    if ok:
        cites = ", ".join(used)
        return {**base, "answer": f"{answer} [{cites}]", "mode": "ghost", "used": used}
    return {**base, "answer": verbatim_fallback(cands), "mode": f"verbatim({why})", "used": [cands[0]["id"]]}

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.environ.get("SEANCE_MODEL", os.path.join(here, "models", "gemma-3-1b-it-Q4_K_M.gguf"))
    server = os.environ.get("SEANCE_LLAMA_SERVER", "llama-server")
    if not os.path.exists(model_path):
        sys.exit(f"no seance without the ghost: model not found at {model_path}\n"
                 "see README.md - any small open-weight instruct GGUF works.")
    from ghost_model import GhostModel
    ghost = GhostModel(model_path, llama_server=server)
    record = load_record()
    print(f"* the candles are lit. the ghost of {record['spirit']} is listening. (Ctrl-C to end)")
    try:
        while True:
            try:
                q = input("\nyou: ").strip()
            except EOFError:
                break
            if not q:
                continue
            r = conduct(q, record, ghost)
            print(f"\nghost: {r['answer']}")
    except KeyboardInterrupt:
        pass
    finally:
        ghost.stop()
    print("\n* the candles go out.")

if __name__ == "__main__":
    main()
