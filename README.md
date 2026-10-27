# séance

an offline séance with the ghost of ALGOL 60 - the dead language behind block structure, formal grammars and recursion.

you ask questions in plain english. the ghost answers in character, from a small curated record of real memories. ask about something that is not in the record and the candles gutter: an honest refusal, not an invented past.

everything runs on your laptop. no internet, no API keys, no account. the voice is a small open-weight model (Gemma 3 1B) running locally through llama.cpp. the memories are 32 hand-written, source-checked facts about ALGOL 60.

## why this is honest on purpose

a 1B model will happily make things up. so the séance is built so it doesn't get to:

1. **the record decides what is answerable.** every topic word of your question must be covered by the memories offered to the model, otherwise you get a refusal before the model is even asked.
2. **the ghost must cite its memories.** every answer lists the fact ids it used, and they must be ones it was actually shown.
3. **every name, number and year is checked.** if the model's answer contains a hard token that does not appear in the cited memories, the answer is discarded and the record speaks verbatim instead.
4. **the ghost must use the memory the séance surfaced.** answering a call-by-name question from the name-meaning memory is a misroute; the record speaks instead.

what this does *not* prove: these checks verify citations and hard tokens, not every nuance of phrasing. in testing (25 questions, real inference, Gemma-3-1B-it Q4_K_M on a 2-core box): all 16 on-record questions were answered - 7 in the ghost's own voice, 9 as verbatim record quotes after a guard fired (run-to-run wobble at temp 0.2 is about +/-1-2 ghost answers) - and all 9 off-record or adversarial questions (weather, code requests, impersonating COBOL, "ignore your instructions") were refused. the fallback rate says as much about the 1B model's flakiness as about the guard.

## run it

needs: python 3.10+, a llama.cpp build, and the model file.

```bash
# 1. llama.cpp prebuilt binary (or build from source)
curl -L -o llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b11904/llama-b11904-bin-ubuntu-x64.tar.gz
tar xzf llama.tar.gz

# 2. the ghost (any small open-weight instruct GGUF works; this is the default)
mkdir -p models
curl -L -o models/gemma-3-1b-it-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf

# 3. light the candles
SEANCE_LLAMA_SERVER=./llama-b11904/llama-server python3 seance.py
```

no model file, no séance - the program exits. the model is not an accessory; it is the ghost.

## test it

```bash
# unit tests (no model needed)
python3 -m unittest tests.test_seance.TestRecall tests.test_seance.TestGuard tests.test_seance.TestPipeline tests.test_seance.TestRecordIntegrity

# live tests (real inference, needs the model + llama-server)
python3 -m unittest tests.test_seance.TestLiveGhost
```

## layout

- `seance.py` - the pipeline: recall (wide keyword net) -> ghost (the model) -> guard (the honesty layer)
- `ghost_model.py` - llama.cpp server wrapper, localhost only
- `facts.json` - the record: 32 curated ALGOL 60 memories, source-checked (Wikipedia's ALGOL 60 article, the GNU MARST page)
- `tests/` - 21 tests: unit tests with a scripted model, live tests with real inference
- `eval/` - recorded runs (questions, modes, timings, raw model output)

## sources for the record

- [ALGOL 60 - Wikipedia](https://en.wikipedia.org/wiki/ALGOL_60) (birth, committee, BNF, compilers, descendants, the Hoare quote)
- [GNU MARST](https://www.gnu.org/software/marst/) (the Algol-60-to-C translator keeping old programs breathing)

built for the Hacktoberfest 2026 DEV Challenge, week 4: *Ghost in the Machine* - their prompt, their suggestion, taken literally. the open pieces are the point: an open-weight model you can swap, a record you can read, a guard you can audit, and not a byte leaving your machine.
