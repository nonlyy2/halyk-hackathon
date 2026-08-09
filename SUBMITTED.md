# Submission v2 — the answers sent, and the code that produced them

`submission.v2-gptoss.json`, built from this exact tree. Supersedes
[`submission/v1-gptoss`](../../tree/submission/v1-gptoss).

| | |
|---|---|
| dataset | private set, 27 scenarios, 84 cells |
| provider | HuggingFace router |
| `complex` (spec, bind) | `openai/gpt-oss-120b:cerebras` |
| `small` (classify, categorise) | `google/gemma-4-31B-it:cerebras` |
| votes | 3 on enrich, spec and bind |
| result | 84/84 cells, all template keys, all well-typed; 26 BREACH, 58 COMPLIANT, 54 with an evidence id |

```bash
export COVENANT_DATA=/path/to/private-set COVENANT_BACKEND=huggingface
export COVENANT_MODEL_SMALL=google/gemma-4-31B-it:cerebras
export COVENANT_MODEL_COMPLEX=openai/gpt-oss-120b:cerebras
uv run covenant run --team "дримтим" --contact-email assylkhan.kerey@nu.edu.kz
```

A fresh `spec` + `bind` on the same model reproduces the same cells exactly — measured on four
scenarios, 13/13 identical. `--votes 3` is what makes that true; at one vote the category split
moves between runs and the answers move with it.

## What changed from v1, and why

Six corrections. Each was verified either by reading the clause, or by re-running the public set
(where a key exists) and confirming both baselines — 33.500 on the original and 32.200 on the
perturbed copy — did not move, so the change only fires where the previous answer was indefensible.

| cell | v1 | v2 | reason |
|---|---|---|---|
| `G2.6.2` | BREACH 268,622,920.31 | BREACH 4,406,389.00 | a bound cost term sixty times the borrower's revenue |
| `H5.6.2` | BREACH 211.19 | COMPLIANT 2.36 | same, against a limit of 2.28 |
| `J2.6.2` | COMPLIANT 3.30 | BREACH 3.30 | the comparison was inverted; 3.30 against a 3.00 cap read as compliance |
| `J4.5.2` | COMPLIANT 64.39 | BREACH 64.39 | "-64.39 <= 3.00" passed a cap on a negative metric |
| `J4.5.3` | BREACH 9,215,956.22 | COMPLIANT 0.30 | a ratio metric whose formula computed money |
| `X3.6.4` | BREACH 918,447.23 | COMPLIANT 918,447.23 | springing covenant: the Group leverage trigger of 3.40x is not met, so the limit does not apply. `actual` stays the true value, as the case requires |

## Known and unfixed

- **`G2.6.1`** reports capital expenditure; the clause measures assets disposed of to an
  Unrestricted Subsidiary. Different quantity. Not corrected: the spec cache is keyed by model and
  the provider ran out of credit before it could be regenerated, and hand-editing one cell would
  produce a file the pipeline cannot reproduce.
- **`J6.6.3`** divides by capital expenditure; the clause defines the metric as a plain sum of
  interest and rent. Same reason.
- **`B2.6.1`, `H2.6.1`** are conditional ("if leverage exceeds 3.00x, then ..."); the condition was
  not modelled, so a breach is reported unconditionally. Eight independent samples failed to pick
  the condition up. Left alone rather than guessed at: asserting compliance without evidence costs
  the same as asserting breach.
- **57 of 67 foreign-currency rows** carry no disclosed rate anywhere in the documents and are
  summed at face value; five cells contain one. No rate was invented, because none is stated.

## Cross-model agreement

An independent run of `spec` + `bind` on `gemini-3.1-flash-lite`, over the same enrichment, agrees
on 60 of 84 cells outright and disagrees on status in 13. Where the disagreement was about a
threshold — a number printed in the clause — reading settled it in this submission's favour every
time (`B3.6.1`, `J2.6.1`, `X2.6.3`, `B5.6.1`). Where it was about which quantity the clause names,
it settled against, in the two cells listed above as unfixed.
