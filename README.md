# covenant-agent

**English** · [Русский](README.ru.md)

Reads "dirty" corporate-loan paperwork — credit agreements, KYC dossiers, audit reports, mixed with
decoys and superseded editions under opaque hash filenames — together with a category-less
transaction ledger, and decides for every financial covenant whether it is **COMPLIANT** or
**BREACH**, what the constrained metric's **actual** value is, and which single transaction is the
**evidence** where one exists.

The deliverable is one `submission.json` matching `submission_template.json`.

## Setup

```bash
uv sync
cp .env.example .env                     # set one API key
brew install tesseract tesseract-lang    # macOS; OCR fallback for scanned pages
export COVENANT_DATA=/path/to/dataset    # or pass --data to every command
uv run covenant ping                     # does the configured provider answer?
```

Any of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`,
`ALIBABA_CLOUD_API_KEY` selects its provider on its own. Every provider is called over plain
`httpx`, so there is no SDK to install and no import that can fail. Keys are re-read from `.env` on
each request, so an exhausted one can be swapped mid-run without restarting a stage.

`ping` makes one call per tier and prints the model and latency, or says exactly what went wrong.
Run it before anything else: from inside a running stage, a bad credential is indistinguishable
from a broken pipeline.

## Running

```bash
uv run covenant run --team TEAM --contact-email you@example.com
```

Every stage in order — classify, enrich, spec, bind, build, doctor. A stage that fails does not
stop it: `build` degrades rather than refusing, so reaching it with partial artefacts beats
submitting nothing, and the failed stage is named at the end so it can be re-run alone.

```bash
uv run covenant classify              # documents -> type + authority
uv run covenant enrich --votes 3      # ledger rows -> categories, tags, audit overrides
uv run covenant spec   --votes 3      # clauses -> formula, threshold, period, carve-out
uv run covenant bind   --votes 3      # covenant term -> the transactions that constitute it
uv run covenant doctor                # preflight: what could still be quietly wrong
uv run covenant build --team TEAM --contact-email you@example.com
```

Every stage caches under `<data>/.cache/` and skips finished scenarios, so a stage that dies
half-way costs only what it had not done. `build` rewrites `submission.json` after each scenario —
there is a valid file on disk from the first scenario onward — and validates it at the end, ending
with either `submittable: N cells, all template keys present and well-typed` or `UNSUBMITTABLE`.
That check is the only one in the pipeline with no judgement in it, so every failure it reports is
certain rather than suspected.

Caches are keyed by model, so switching models never reuses another model's judgement.

`--votes 3` is not optional: at one vote the category split moves between runs and the answer moves
with it; at three, two consecutive runs produce byte-identical cells.

Do **not** pass `--review`. Measured on a flash-class model it cost 7.6 points (33.0 → 25.4): asked
"is anything wrong here", a small model finds something. Its guardrails held — every correction was
recomputed and none could assert a status or a value — so the failure was selection quality, not
safety. Re-measure before enabling it on a stronger model, and only against a key.

## How it works

**Language models classify, extract and select. Python computes.** Models decide *which* ledger
rows count and *how* they combine; they never do arithmetic. `actual` is scored on a scale that
reaches zero at 5% relative error, so a silent arithmetic slip costs as much as no answer at all.
Every sum, formula evaluation and comparison happens in deterministic Python over a restricted AST
— never `eval` on unchecked model output.

Numbers come from the ledger, except the ones documents authoritatively supply or override: an
auditor's reclassification, a disclosed FX pair, a corrected or missing amount, a provision
disclosed but never transacted, a parent's group figure.

```
src/covenant/
  config.py              every path resolved once, from --data or COVENANT_DATA
  cli.py                 the entrypoint: classify / enrich / spec / bind / build / doctor / ping / score
  llm/client.py          provider selection, model fallback chains, live key reload
  ingest/
    documents.py         pypdfium2 text, per-page OCR fallback, cached, serialised (pdfium is not thread-safe)
    ledger.py            txn_id prefix -> scenario_id; account_id <-> scenario_id from the data
    matching.py          documents -> scenarios, by account id then by borrower name
    textnorm.py          homoglyph/whitespace normalisation for letter-spaced PDF text
  analyze/
    classify.py          document type, and whether it speaks for the covenants
    clauses.py           slices clause text, KYC ownership, audit addendum, with fallbacks
    documents_index.py   the single "which documents speak for this borrower" lookup
    enrich.py            per-transaction category, related-party tags, audit overrides
    spec.py              clause -> computable spec (roles, formula, threshold, carve-out)
    binding.py           covenant term -> the transactions that actually constitute it
    review.py            optional second opinion on fragile cells (off by default, see above)
  scoring/
    compute.py           the deterministic evaluator
    confidence.py        per-cell fragility signals, for triage
    rubric.py            dev-only scorer implementing the case's own scale
tools/perturb.py         builds a perturbed dataset to test generalisation
```

### Where a covenant term comes from

A clause names a line item — "Операционные расходы по аудированной отчётности Заёмщика" — and the
ledger plants far larger rent, payroll, marketing and insurance rows beside the real one, all of
which a textbook chart of accounts would also call operating costs. Bucketing by accounting
substance reads the term an order of magnitude too wide, and the metric with it.

So membership is decided by `binding.py`, as a reading task over the actual transactions with the
clause's wording in view, and it is asked with the clause's **threshold masked** — selecting rows
and knowing the number they must clear are two jobs that must not meet. Two terms are never bound:
the related-party and unrestricted-subsidiary aggregates follow from the KYC table by exact name
match, because the dataset seeds lexically similar decoy counterparties to defeat anything looser.

Deterministic guards sit under it, all model-independent. A selection covering more than half the
ledger is not a line item; a cost term totalling many times revenue is not one either; a ratio whose
denominator resolves to exactly its numerator's rows is an artifact; a ratio metric a thousand times
its own limit computed money, not a ratio; a cap tested against a negative metric has not been
satisfied. A clause granting a permitted basket states two numbers and only the first is the limit —
the basket belongs inside the metric, subtracted and capped, and a model that also records it as a
carve-out applies it twice. `doctor` says whenever a guard overruled the model's own selection,
because these are the lines most able to cap how right a stronger model is allowed to be.

## Dev scoring

With `ground_truth.json` present:

```bash
uv run covenant build --team dev --contact-email dev@example.com --score
```

Applies the case's own rubric — status gating at 0.50, `actual` decaying to zero at 5%, evidence
worth 0.20 and riding on `actual` where the key holds no transaction id. It reports an unweighted
mean; the real leaderboard weights cells by an undisclosed complexity factor, so treat it as a logic
check, not a prediction. **The production path never reads `ground_truth.json`.**

## Generalisation

The pipeline carries no hardcoded scenario ids, thresholds, currencies, company names or covenant
numbers. Prompts are principle-based, the category taxonomy is derived at runtime, and thresholds
and definitions are read from each document.

Because a 36-cell public set with a visible key is a poor judge of whether a change generalises,
`tools/perturb.py` builds the nearest thing to a held-out set: it changes what an unseen set could
plausibly do differently while keeping the answer derivable from the old key.

```bash
uv run python tools/perturb.py reformat --data data --out /tmp/pert  # article number, clause captions, headings
uv run python tools/perturb.py relabel  --data data --out /tmp/pert  # scenario, account, txn ids, row order
uv run python tools/perturb.py rescale  --data data --out /tmp/pert  # every amount and $ threshold x1000
```

Under `reformat`, all 12 public scenarios still located and sliced every clause, the KYC ownership
section still resolved, and 11 of 12 enriched identically — the one difference being a category
named in Russian rather than English. The score is 32.2 against 33.5 on the original, and that gap
is the honest reading of how much of the 33.5 is robust: two cells move because a bound term
absorbed rows it should not have, though the clause text and every transaction were identical.
Term boundaries are the real uncertainty, not formatting, arithmetic or plumbing.

**Raising `--votes` does not fix that, measured.** At `--votes 5`, 50 of 51 bound terms come back
identical to `--votes 3` and both datasets score the same. Sampling noise is exhausted — the model
answers the perturbed input confidently and differently.

The remaining dataset-specific literals are recurring format-convention strings (heading names,
supersession markers) used as slicers, each with a general fallback beneath it, and `doctor` reports
whenever a fallback fired.

---

# Operating this on an unseen set

There is no key on a graded set. Nothing during the run can tell you whether an answer is right, so
the score cannot be chased there. What *can* be fixed is mechanical failure: a clause whose text was
never found, a KYC dossier that yielded no owners, a term nothing bound, a currency left
unconverted, a stage that died. Every one costs whole cells and every one is visible without a key.
`covenant doctor` is the instrument.

Measured on a 27-scenario, 84-cell set: document extraction 10s, `classify` ~5 min,
`enrich --votes 3` ~12 min, `spec` and `bind` ~5 min each at `COVENANT_CONCURRENCY=4`. Time is not
the constraint; diagnosing a stage that fails is.

## What `doctor` reports, ranked by what it costs to ignore

| Line | Meaning | Act on it? |
|---|---|---|
| `no authoritative credit agreement` | Those cells have no covenant text at all | **Yes — highest.** Check `classify` did not mark every edition superseded |
| `no spec cache` / `no enriched cache` | A stage failed for that borrower | **Yes.** Re-run the stage for that scenario alone |
| `unresolvable formula variables` | The formula names something nothing populated | **Yes.** Re-run `spec <sid> --fresh` |
| `clause headings not found` | That covenant is answered from a fallback, not its own wording | **Yes.** Open the agreement and see how it captions clauses |
| `the spec calls its own formula approximate` | The model wrote "used as a proxy" / "cannot be evaluated" in its own notes | **Yes.** It is confessing; read the worksheet |
| `a guard discarded the model's own selection` | A deterministic guard overruled the binder | Read the worksheet — right when the model over-widened, wrong when it was reading a genuinely broad base |
| `KYC ownership heading not found` | Fell back to the whole dossier — usually still works | Only if a related-party covenant also reports zero matches |
| `unbound terms [...]` | The binder did not answer for a term; role map used | Low. Correct in many scenarios |
| `confidence=low/medium` | Thin margin, or one row deciding the cell | **Not actionable without a key.** Information, not a task |

The last row matters most for what it forbids. A thin margin is not evidence of an error, and
"fixing" a cell because its margin is thin means changing an answer with no evidence it is wrong.

## Reading a worksheet

```bash
python -c "import json;w=json.load(open('$COVENANT_DATA/.cache/worksheets.json'));print(json.dumps(w['<sid>']['<key>'],ensure_ascii=False,indent=2))"
```

Per cell: the formula, every term, which rows fed it and the subtotal — the only artefact that shows
*why* a number came out as it did. Two failures are visible here and nowhere else: a term holding
one row when the caption plainly spans several (or the reverse), and an inflow counted as revenue
that is not a sale. Both are judgements about the documents, so they can be made without a key. If a
term is plainly wrong, fix it in `bind`'s prompt as a *principle* and re-run `bind` — never patch
one scenario by hand, and never edit `submission.json` directly.

## When something goes wrong

| Symptom | Do |
|---|---|
| A stage dies on one scenario | It is caught per-scenario; the others finish. Re-run that stage with the scenario id as an argument |
| Quota exhausted mid-run | The client falls through to the next model in `COVENANT_MODEL_*` and remembers the dead one for ten minutes. Add more names and re-run — caches make it resume |
| A key is replaced mid-run | Picked up on the next request; the exhausted-model state is cleared with it |
| Nothing classified as authoritative | Check for a supersession marker matching every edition. `<data>/.cache/classifications.jsonl` gives the reason per document |
| Running out of time | Stop at whatever stage is done and run `build`. It degrades: no binding → role map; no spec → the covenant's own threshold as `actual`. Never submit nothing |

**Never** run `--fresh` on a stage that has already succeeded unless you mean to spend its full cost
again. **Never** hand-edit `submission.json`. **Never** delete the cache directory.

---

# What was submitted

Three answers were sent for a private set of 27 scenarios and 84 cells. Each is archived on its own
branch and tag alongside the code that produced it.

| | file | branch / tag | `complex` model | `small` model |
|---|---|---|---|---|
| v1 | `submission.v1-gptoss.json` | `submission/v1-gptoss`, `submitted-v1` | `openai/gpt-oss-120b:cerebras` | `google/gemma-4-31B-it:cerebras` |
| v2 | `submission.v2-gptoss.json` | `submission/v2-gptoss`, `submitted-v2` | `openai/gpt-oss-120b:cerebras` | `google/gemma-4-31B-it:cerebras` |
| v3 | `submission.v3-gemini.json` | `submission/v3-gemini`, `submitted-v3` | `gemini-3.1-flash-lite` | `gemini-3.1-flash-lite` |

All used `--votes 3`. Enrichment is shared, so v3 differs from v2 only in the model that produced
the covenant specs and the term bindings. A fresh `spec` + `bind` on the same model reproduces the
same cells exactly — measured on four scenarios, 13/13 identical.

## Model benchmark, measured against the public key

Every model ran the full `spec` + `bind` on the public set with the same shared enrichment, so the
only variable is the model on the `complex` tier.

| `complex` model | public 36 | perturbed 36 |
|---|---|---|
| `gemini-3.1-flash-lite` | **33.5** | **32.2** |
| `qwen3.7-max` | **33.5** | not measured |
| `qwen3-max` | **33.5** | not measured |
| `openai/gpt-oss-120b:cerebras` | 32.5 | **32.2** |
| `qwen3.6-max-preview` | 32.5 | not measured |
| `deepseek-v4-pro` | 32.2 | not measured |
| `glm-5.2` | 31.5 | not measured |

**Model strength does not predict score here.** The smallest and cheapest model ties for the lead,
and two far larger ones lose to it. The prompts were written while watching the failure modes of
flash-class models, and a stronger model reads the same instructions differently. Before adopting
any model, run `covenant build --score` on the public set — five minutes, and it answers what
guessing cannot.

## Known-wrong cells

Verified by reading the clause text, cached in `<data>/.cache/specs/*.json` under `"clauses"`.
Neither version dominates; each is right where the other is wrong.

| cell | v2 | v3 | what the clause says |
|---|---|---|---|
| `B5.6.1` | ✅ | ✗ | total debt is "financing drawn during the period"; v3 uses a balance-sheet figure |
| `H3.6.2` | ✅ | ✗ | exceeding the rent cap is excused when insurance ≥ $200,000, and it is ($236,904) |
| `G2.6.1` | ✗ | ✅ | measures assets disposed of to an Unrestricted Subsidiary, not capital expenditure |
| `G2.6.2` | ✗ | ✅ | the metric is EBITDA **plus** financing receipts; v2 has only EBITDA |
| `J6.6.3` | ✗ | ✅ | "означают **сумму**" — v2 invented a division |
| `H2.6.1` | ✗ | ✅ | springing on leverage above 3.00x; v2 reports a breach unconditionally |
| `B3.6.1`, `X2.6.3`, `J2.6.1` | ✅ | ✅ | both correct after the permitted-basket fix |

Unresolved in both, and not guessed at:

- **`B2.6.1`** is springing on a leverage ratio defined "согласно Пункту 5.1", and article 5 is not
  extractable from that agreement. The trigger cannot be computed, so neither verdict can be
  established. Both report a breach.
- **`J4.5.3`** caps revenue in any single quarter as a fraction of the year's. The formula language
  has no "maximum over quarters", so the covenant is reported as uncomputable and falls back to the
  clause's own limit.
- **57 of 67 foreign-currency rows** carry no disclosed rate anywhere in the documents and are
  summed at face value; five cells contain one. No rate was invented, because none is stated.
