# Submission runbook — the three-hour window

**There is no key on the private set.** Nothing in that window can tell you whether an answer is
right, so the score cannot be optimised there and must not be chased. What *can* be fixed is
mechanical failure: a clause whose text was never found, a KYC dossier that yielded no owners, a
term nothing bound, a currency left unconverted, a stage that died. Every one of those costs whole
cells and every one is visible without a key. `covenant doctor` is the instrument; this document is
the order of operations.

Budget the window as **90 minutes of pipeline, 60 minutes of reading `doctor`, 30 minutes spare.**

---

## 0. Before the data arrives (do this now, not then)

```bash
uv sync
uv run pytest -q                      # 55 tests, all must pass
tesseract --version                   # OCR fallback for scanned pages
```

Have the `.env` ready with the strongest model available on the `complex` tier and a chain of
alternates behind it, so one exhausted quota does not stop the run:

```
COVENANT_BACKEND=gemini
COVENANT_MODEL_SMALL=<fast model>,<alternate>,<alternate>
COVENANT_MODEL_COMPLEX=<strongest model>,<alternate>,<alternate>
COVENANT_CONCURRENCY=4
COVENANT_MIN_INTERVAL=4.5             # 0 on a paid endpoint
```

`complex` carries `spec` and `bind` — the two stages that decide what a covenant measures and which
rows it measures. That is where model quality converts into score.

---

## 1. Run (about 25 minutes on a flash-class model, 4-way concurrent)

```bash
export COVENANT_DATA=/path/to/private-set

uv run covenant classify              # ~68 documents; cached, resumable
uv run covenant enrich --votes 3      # categories, related-party tags, audit overrides
uv run covenant spec   --votes 3      # clause -> formula, threshold, period, carve-out
uv run covenant bind   --votes 3      # covenant term -> the transactions that constitute it
uv run covenant build --team "дримтим" --contact-email assylkhan.kerey@nu.edu.kz
```

Every stage caches and is safe to re-run: finished scenarios are skipped, so a stage that dies
half-way costs only what it had not done. `build` rewrites `submission.json` after every scenario,
so **there is a valid submission on disk from the first scenario onward.** Produce one early and
keep producing them; never leave the last write to the end of the window.

`--votes 3` is not optional. At one vote the same ledger comes back with a different category split
each run and the answer moves with it; at three, two consecutive runs produced byte-identical cells.

Do **not** pass `--review`. Measured on a flash-class model it cost 7.6 points (33.0 → 25.4): asked
"is anything wrong here", a small model finds something. Re-measure before trusting it on a stronger
one, and only if there is a key to measure against.

---

## 2. Read `doctor` (this is where the window is actually spent)

```bash
uv run covenant doctor
```

Ranked by what it costs if ignored:

| Report line | What it means | Worth acting on? |
|---|---|---|
| `no authoritative credit agreement` | Three cells have no covenant text at all | **Yes — highest.** Check `classify` didn't mark every edition superseded |
| `clause headings not found: [...]` | That covenant will be answered from the fallback, not its own wording | **Yes.** Open the agreement, see how it captions clauses |
| `no spec cache` / `no enriched cache` | That stage failed for this borrower | **Yes.** Re-run the stage for that scenario alone |
| `unresolvable formula variables` | The formula names something nothing populated | **Yes.** Re-run `spec <sid> --fresh` |
| `KYC ownership heading not found` | Fell back to the whole dossier — usually still works | Only if a related-party covenant also reports zero matches |
| `term(s) too wide to be a line item` | The binder swept the ledger; the role map is being used instead | Read the worksheet — often already correct |
| `unbound terms [...]` | The binder didn't answer for a term; role map used | Low. Correct in many scenarios |
| `confidence=low/medium` | Thin margin or one row deciding the cell | **Not actionable without a key.** Information, not a task |

The last row matters most in what it forbids. A thin margin is not evidence of an error, and
"fixing" a cell because its margin is thin means changing an answer with no evidence it is wrong.
Do not do it.

## 3. Read the worksheets

```bash
python -c "import json;w=json.load(open('$COVENANT_DATA/.cache/worksheets.json'));print(json.dumps(w['<sid>']['<key>'],ensure_ascii=False,indent=2))"
```

Per cell: the formula, every term, which rows fed it and the subtotal. This is the only artefact
that shows *why* a number came out as it did. Two failures are visible here and nowhere else:

- **a term holding one row when the caption plainly spans several**, or the reverse;
- **an inflow counted as revenue that is not a sale** — a refund, rebate, sublet rent, interest
  income, a deposit returned.

Both are judgements about the documents, so they can be made without a key. If a term is plainly
wrong, fix it in `bind`'s prompt as a *principle* and re-run `bind` for all scenarios — never patch
one scenario by hand, and never edit `submission.json` directly.

---

## 4. Before submitting

`build` validates its own output and ends with one of two lines:

```
submittable: 36 cells, all template keys present and well-typed
UNSUBMITTABLE -- 2 problem(s): ...
```

A missing, renamed or mistyped cell scores zero exactly like a wrong one, and the difference is
invisible in a 36-cell file read by eye under time pressure. If that line does not say
`submittable`, fix it before anything else — it is the only check in the whole pipeline with no
judgement in it, so every failure it reports is certain rather than suspected.

---

## If something goes wrong

| Symptom | Do |
|---|---|
| A stage dies on one scenario | It is caught per-scenario; the others finish. Re-run that stage with the scenario id as an argument |
| Quota exhausted mid-run | The client falls through to the next model in the chain automatically and remembers the dead one. Add more names to `COVENANT_MODEL_*` and re-run — caches make it resume |
| Nothing classified as authoritative | Check for a supersession marker matching every edition. `data/.cache/classifications.jsonl` shows the reason per document |
| Running out of time | Stop at whatever stage is done and run `build`. It degrades: no binding → role map; no spec → the covenant's own threshold as `actual`. Never submit nothing |

**Never** run `--fresh` on a stage that has already succeeded unless you intend to spend its full
cost again. **Never** hand-edit `submission.json`. **Never** delete the cache directory.
