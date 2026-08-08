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
    clauses.py           slices clause text, KYC ownership, audit addendum
    documents_index.py   the single "which documents speak for this borrower" lookup
    enrich.py            per-transaction category, related-party & unrestricted-sub tags, overrides
    spec.py              clause -> computable spec (roles, formula, threshold, carve-out)
  scoring/
    compute.py           the deterministic evaluator
    confidence.py        per-cell fragility signals, for triage
    rubric.py            dev-only scorer implementing the case's own scale
```

## Setup

```bash
uv sync
cp .env.example .env          # add one backend key
brew install tesseract tesseract-lang    # macOS; OCR fallback needs the rus+eng packs
export COVENANT_DATA=/path/to/dataset    # or pass --data to every command
```

## Running

```bash
covenant classify            # documents -> type + authority          (cached)
covenant enrich              # ledger rows -> categories, tags, overrides
covenant spec --votes 3      # clauses -> computable specs, self-consistency voted
covenant build --team TEAM --contact-email you@example.com
```

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
