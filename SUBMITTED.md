# Submission v1 — the answers actually sent, and the code that produced them

`submission.v1-gptoss.json`, built from this exact tree.

| | |
|---|---|
| dataset | private set, 27 scenarios, 84 cells |
| provider | HuggingFace router |
| `complex` (spec, bind) | `openai/gpt-oss-120b:cerebras` |
| `small` (classify, categorise) | `google/gemma-4-31B-it:cerebras` |
| votes | 3 on enrich, spec and bind |
| result | 84/84 cells, all template keys, all well-typed; 27 BREACH, 57 COMPLIANT, 53 with an evidence id |

Reproduce with:

```bash
export COVENANT_DATA=/path/to/private-set COVENANT_BACKEND=huggingface
export COVENANT_MODEL_SMALL=google/gemma-4-31B-it:cerebras
export COVENANT_MODEL_COMPLEX=openai/gpt-oss-120b:cerebras
uv run covenant run --team "дримтим" --contact-email assylkhan.kerey@nu.edu.kz
```

## What was known to be wrong at the time of sending

Four cells were later found to assert numbers they could not mean, and are corrected on the
mainline after this branch:

| cell | v1 said | corrected to | why v1 was wrong |
|---|---|---|---|
| `G2.6.2` | BREACH 268,622,920.31 | BREACH 4,406,389.00 | a bound cost term totalling far more than revenue |
| `H5.6.2` | BREACH 211.19 | COMPLIANT 2.36 | same, against a limit of 2.28 |
| `J4.5.2` | COMPLIANT 64.39 | BREACH 64.39 | "-64.39 <= 3.00" passed a cap on a negative metric |
| `J4.5.3` | BREACH 9,215,956.22 | COMPLIANT 0.30 | a ratio metric whose formula computed money |

Known and unfixed in both versions: 57 of 67 foreign-currency rows carry no disclosed rate
anywhere in the documents and are summed at face value; five cells contain one. No rate was
invented, because none is stated.
