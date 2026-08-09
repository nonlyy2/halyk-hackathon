# covenant-agent

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
`httpx`, so there is no SDK to install. Keys are re-read from `.env` on each request, so an
exhausted one can be swapped mid-run.

## Running

```bash
uv run covenant run --team TEAM --contact-email you@example.com
```

That is every stage in order — classify, enrich, spec, bind, build, doctor — and it is what to use
under time pressure. A stage that fails does not stop it: `build` degrades rather than refusing, so
reaching it with partial artefacts beats submitting nothing. The stages also exist separately, for
re-running just one:

```bash
uv run covenant classify              # documents -> type + authority
uv run covenant enrich --votes 3      # ledger rows -> categories, tags, audit overrides
uv run covenant spec   --votes 3      # clauses -> formula, threshold, period, carve-out
uv run covenant bind   --votes 3      # covenant term -> the transactions that constitute it
uv run covenant doctor                # preflight: what could still be quietly wrong
uv run covenant build --team TEAM --contact-email you@example.com
```

Every stage caches under `<data>/.cache/` and skips finished scenarios, so a stage that dies
half-way costs only what it had not done. `build` rewrites `submission.json` after each scenario
and validates it at the end. Caches are tagged with the model that produced them, so switching
models never reuses another model's judgement.

`--votes 3` is not optional: at one vote the category split moves between runs and the answer moves
with it; at three, two consecutive runs produce identical cells.

See **[RUNBOOK.md](RUNBOOK.md)** for the timed submission window, where there is no answer key and
only mechanical failures can be fixed.

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
    documents.py         pypdfium2 text, per-page OCR fallback, cached, parallel
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
    review.py            optional second opinion on fragile cells (off by default, see below)
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

Two deterministic guards sit under it, both model-independent: a selection covering more than half
the ledger is not a line item and falls back to the role map, and a ratio whose denominator
resolves to exactly its numerator's rows is an artifact, not a measurement.

### `--review` is off by default

The critic pass was measured on `gemini-3.1-flash-lite` and cost 7.6 points (33.0 → 25.4): it
changed 14 of 36 cells and most had been correct. Asked "is anything wrong here", a small model
finds something. Its guardrails held — every correction was recomputed and none could assert a
status or a value — so the failure was selection quality, not safety. Re-measure before enabling it
on a stronger model.

## Dev scoring

With `ground_truth.json` present:

```bash
uv run covenant build --team dev --contact-email dev@example.com --score
```

Applies the case's own rubric — status gating at 0.50, `actual` decaying to zero at 5%, evidence
worth 0.20 and riding on `actual` where the key holds no transaction id. It reports an unweighted
mean; the real leaderboard weights cells by an undisclosed complexity factor, so treat it as a
logic check, not a prediction. **The production path never reads `ground_truth.json`.**

## Generalisation

The private set shares the public set's formats but not its values, so the pipeline carries no
hardcoded scenario ids, thresholds, currencies, company names or covenant numbers. Prompts are
principle-based, the category taxonomy is derived at runtime, and thresholds and definitions are
read from each document.

Because a 36-cell public set with a visible key is a poor judge of whether a change generalises,
`tools/perturb.py` builds the nearest thing to a held-out set: it changes what the private set
could plausibly do differently while keeping the answer derivable from the old key.

```bash
uv run python tools/perturb.py reformat --data data --out /tmp/pert  # article number, clause captions, headings
uv run python tools/perturb.py relabel  --data data --out /tmp/pert  # scenario, account, txn ids, row order
uv run python tools/perturb.py rescale  --data data --out /tmp/pert  # every amount and $ threshold x1000
```

Under `reformat`, all 12 scenarios still located and sliced every clause, the KYC ownership section
still resolved, and 11 of 12 enriched identically — the one difference being a category named in
Russian rather than English. The remaining dataset-specific literals are recurring format-convention
strings (heading names, supersession markers) used as slicers, each with a general fallback beneath
it, and `doctor` reports whenever a fallback fired.

The score under `reformat` is 32.2 against 33.5 on the original, and the gap is the honest reading
of how much of that 33.5 is robust. Two cells move, both because a bound term absorbed rows it
should not have — cosmetic changes to how the documents are captioned shifted the row selection,
though the clause text and every transaction were identical. Term boundaries are therefore the
pipeline's real uncertainty, not its formatting, its arithmetic or its plumbing.

**Raising `--votes` does not fix that, measured.** At `--votes 5`, 50 of 51 bound terms come back
identical to `--votes 3` and both datasets score exactly the same; the single term that moves is
one the line-item guard discards either way. Voting suppresses sampling noise, and there is none
left to suppress — the model answers the perturbed input confidently and differently. The remaining
lever is model quality on the `complex` tier, not more samples of the same judgement.
