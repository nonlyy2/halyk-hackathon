import re
from dataclasses import dataclass

from covenant.ingest.documents import Doc
from covenant.ingest.ledger import Ledger
from covenant.ingest.textnorm import normalize

_MONEY_RE = re.compile(r"\$\s?\d[\d,\s]*\.\d{2}|\$\d[\d,]{5,}")
_MIN_MONEY_FIGURES = 8  # see match_company_names


@dataclass(frozen=True)
class Match:
    doc_id: str
    scenario_ids: list[str]
    account_ids: list[str]


def match_company_names(
    docs: list[Doc], names_by_scenario: dict[str, str], matched: set[str]
) -> list[Match]:
    """Second pass for documents that carry no account id at all.

    A borrower's group figures can sit in the ultimate parent's consolidated report, which names the
    subsidiary but never quotes an `ACC-` number -- so the account-id pass cannot see it, and the
    covenant that needs it has no source. Matching is on the borrower name taken from its own credit
    agreement, whitespace-normalized, because those names get wrapped across lines in the PDFs
    ("Ekibastuz Power\\nServices JSC") and would otherwise miss on an exact substring test.

    Only documents the first pass did not already claim are considered, so this can add scenarios
    but never re-attribute one.
    """
    needles = {
        sid: normalize(name) for sid, name in names_by_scenario.items() if name and len(name) > 8
    }

    out: list[Match] = []
    for doc in docs:
        if doc.doc_id in matched:
            continue
        # The borrower's name also appears across its internal HR, IT and facilities paperwork,
        # none of which can carry a covenant figure. This pass exists to find FINANCIAL statements,
        # and a financial statement is distinguished from a memo by being full of money: the
        # threshold drops the noise without any judgement about what the documents are called.
        if len(_MONEY_RE.findall(doc.text)) < _MIN_MONEY_FIGURES:
            continue
        haystack = normalize(doc.text)
        found = sorted(sid for sid, needle in needles.items() if needle in haystack)
        if found:
            out.append(Match(doc_id=doc.doc_id, scenario_ids=found, account_ids=[]))
    return out


def match_accounts(docs: list[Doc], ledger: Ledger) -> list[Match]:
    normalized_accounts = {
        account_id: normalize(account_id) for account_id in ledger.account_to_scenario
    }

    matches: list[Match] = []
    for doc in docs:
        haystack = normalize(doc.text)
        found_accounts = sorted(
            account_id for account_id, needle in normalized_accounts.items() if needle in haystack
        )
        if not found_accounts:
            continue
        found_scenarios = sorted({ledger.account_to_scenario[a] for a in found_accounts})
        matches.append(
            Match(doc_id=doc.doc_id, scenario_ids=found_scenarios, account_ids=found_accounts)
        )
    return matches
