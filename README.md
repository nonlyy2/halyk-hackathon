# covenant-agent

Reads "dirty" corporate-loan paperwork — credit agreements, KYC dossiers, audit reports, mixed with
decoys and superseded editions under opaque hash filenames — together with a category-less
transaction ledger, and decides for every financial covenant whether it is **COMPLIANT** or
**BREACH**, what the constrained metric's **actual** value is, and which single transaction is the
**evidence** where one exists.

The deliverable is a single `submission.json` matching `submission_template.json`.

## The one design rule

**Language models classify, extract and structure. Python computes.**

Models decide *which* ledger rows count and *how* they combine; they never do arithmetic. `actual`
is scored on a decaying scale that reaches zero at 5% relative error, so a silent arithmetic slip
is as expensive as not answering at all. Every sum, formula evaluation and comparison happens in
deterministic Python over a restricted AST — never `eval` on unchecked model output.

Numbers come from the ledger, except the specific ones the documents authoritatively supply or
override: an auditor's reclassification, a disclosed FX pair, a corrected or missing amount, a
provision that was disclosed but never transacted, a parent company's group figure.

## Layout

```
src/covenant/
  config.py              every path resolved once, from --data or COVENANT_DATA
  cli.py                 the only entrypoint: classify / enrich / spec / build / score
  llm/client.py          backend selection (Anthropic, Gemini, HF, Alibaba, local Ollama)
  ingest/
    documents.py         pypdfium2 text, per-page OCR fallback for scanned pages, cached
    ledger.py            txn_id prefix -> scenario_id; account_id <-> scenario_id from the data
    matching.py          documents -> scenarios, by account id and then by borrower name
    textnorm.py          homoglyph/whitespace normalisation for letter-spaced PDF text
  analyze/
    classify.py          document type, and whether it speaks for the covenants
    clauses.py           slices clause text, KYC ownership, audit addendum, with fallbacks
    documents_index.py   the single "which documents speak for this borrower" lookup
    enrich.py            per-transaction category, related-party & unrestricted-sub tags, overrides
    spec.py              clause -> computable spec (roles, formula, threshold, carve-out)
    binding.py           covenant term -> the transactions that actually constitute it
    review.py            second opinion on the cells whose own signals say they are fragile
  scoring/
    compute.py           the deterministic evaluator
    confidence.py        per-cell fragility signals, for triage
    rubric.py            dev-only scorer implementing the case's own scale
```

## Where a term comes from

A covenant clause names a line item — "Операционные расходы по аудированной отчётности Заёмщика" —
and the ledger plants far larger rent, payroll, marketing and insurance rows beside the real one,
all of which a textbook chart of accounts would also call operating costs. Bucketing by accounting
substance therefore reads the term an order of magnitude too wide, and the metric with it.

So membership is decided by `binding.py`, as a reading task over the actual transactions with the
clause's wording in view, and it is asked with the clause's **threshold masked** — selecting rows
and knowing the number they must clear are two jobs that must not meet. `spec.py` still decides the
covenant's shape; the model only ever selects rows, and `compute.py` still does every sum.

Two terms are deliberately never bound: the related-party and unrestricted-subsidiary aggregates
follow from the KYC table by exact name match, because this dataset seeds lexically similar decoy
counterparties precisely to defeat anything looser.

## Setup

```bash
uv sync
cp .env.example .env          # add one backend key
brew install tesseract tesseract-lang    # macOS; OCR fallback needs the rus+eng packs
export COVENANT_DATA=/path/to/dataset    # or pass --data to every command
```

## Running

```bash
covenant classify                    # documents -> type + authority          (cached)
covenant enrich --votes 3            # ledger rows -> categories, tags, overrides
covenant spec   --votes 3            # clauses -> computable specs, self-consistency voted
covenant bind   --votes 3            # covenant terms -> transactions, voted per row
covenant doctor                      # preflight: what could still be quietly wrong
covenant build --review --team TEAM --contact-email you@example.com
```

`doctor` reads only artefacts the run produced — never the ground truth — and reports what would
otherwise cost cells silently: a borrower with no authoritative agreement, a clause heading that was
not found, a KYC ownership section that fell back to the whole dossier, an unresolvable formula
variable, an unconverted foreign-currency row, a term nothing bound, a cell sitting on a knife edge.
It is the list to spend the end of a timed run on.

`build` writes `submission.json` after every scenario, so a crash late in a run still leaves a
submittable file, and writes a per-cell worksheet to `<data>/.cache/worksheets.json` — which
transactions fed which term — which is what to read when a cell looks wrong.

### `--review` is off by default, and should stay off on a small model

The critic pass is opt-in because, measured, it made things markedly worse on `gemini-3.1-flash-lite`:
**33.0 → 25.4 / 36**. It changed 14 of the 36 cells and most of those had been correct — asked
"is anything wrong here", a small model finds something. The guardrails held (every correction was
recomputed, none set a status or a number directly), so nothing became invalid; the selections were
just worse. Re-measure it before enabling it on any new model, and only keep it if the score moves
the right way.

Each stage caches under `<data>/.cache/`, so reruns are cheap and every intermediate is
inspectable. Spec caches are tagged with the model that produced them, so switching models writes
separate files instead of silently reusing another model's judgement.

### Dev scoring

With `ground_truth.json` present:

```bash
covenant build --team dev --contact-email dev@example.com --score
```

This applies the case's own rubric — status gating at 0.50, `actual` decaying to zero at 5%,
evidence worth 0.20 and riding on `actual` where the key holds no transaction id — and prints a
per-cell report. It reports an unweighted mean; the real leaderboard weights cells by an
undisclosed complexity factor, so treat it as a logic check, not a prediction. **The production
path never reads `ground_truth.json`.**

## Generalisation

The private set shares the public set's formats but not its values, so the pipeline carries no
hardcoded scenario ids, thresholds, currencies, company names or covenant numbers. Prompts are
principle-based, the category taxonomy is derived at runtime, and thresholds and definitions are
read from each document. The only dataset-specific literals are recurring format-convention strings
— heading names, supersession markers — used as slicers with general-criteria fallbacks.
