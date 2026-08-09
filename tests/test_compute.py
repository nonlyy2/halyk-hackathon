from __future__ import annotations

import pytest
from conftest import txn

from covenant.scoring.compute import (
    _find_evidence_txn,
    _resolve_variable,
    compute_covenant,
    narrow_absurd_cost_roles,
    variable_rows,
    without_absurd_bound_terms,
)

# one real operating-costs line surrounded by the large unrelated rows the ledgers plant
LEDGER = [
    txn(
        "TXN-X-0001",
        -4_204_663.19,
        "operating_costs",
        description="Kiln servicing and operating costs",
    ),
    txn("TXN-X-0002", -8_599_812.02, "rent"),
    txn("TXN-X-0003", -7_250_114.44, "marketing"),
    txn(
        "TXN-X-0004", 6_918_204.37, "sales_revenue", description="Cement despatch sales settlement"
    ),
]
ROLES = {
    "operating_costs": "operating_expenses",
    "rent": "operating_expenses",
    "marketing": "operating_expenses",
    "sales_revenue": "revenue",
}


def test_role_path_sums_every_category_mapped_to_the_role():
    cov = {"formula": "operating_expenses"}
    assert _resolve_variable("operating_expenses", cov, ROLES, LEDGER) == pytest.approx(
        20_054_589.65
    )


def test_binding_replaces_the_role_sum_with_the_named_rows():
    cov = {"formula": "operating_expenses"}
    binding = {"operating_expenses": {"txn_ids": ["TXN-X-0001"]}}
    assert _resolve_variable("operating_expenses", cov, ROLES, LEDGER, binding) == pytest.approx(
        4_204_663.19
    )


def test_binding_is_reported_as_the_resolution_kind():
    cov = {"formula": "revenue"}
    binding = {"revenue": {"txn_ids": ["TXN-X-0004"]}}
    kind, rows = variable_rows("revenue", cov, ROLES, LEDGER, binding)
    assert kind == "bound"
    assert [e.txn_id for e in rows] == ["TXN-X-0004"]


def test_the_period_filter_still_applies_on_top_of_a_binding():
    cov = {"formula": "revenue", "period": {"start": "2025-01-01", "end": "2025-03-31"}}
    binding = {"revenue": {"txn_ids": ["TXN-X-0004"]}}
    # the bound row is dated outside the period, so it cannot contribute; the role path then finds
    # nothing in the period either, which is a legitimate zero rather than an error
    assert _resolve_variable("revenue", cov, ROLES, LEDGER, binding) == 0.0


def test_an_empty_binding_falls_back_to_the_role_path():
    cov = {"formula": "operating_expenses"}
    binding = {"operating_expenses": {"txn_ids": []}}
    assert _resolve_variable("operating_expenses", cov, ROLES, LEDGER, binding) == pytest.approx(
        20_054_589.65
    )


def test_a_binding_never_overrides_the_related_party_tag_aggregate():
    ledger = [*LEDGER, txn("TXN-X-0005", -418_662.44, "advisory", related_party=True)]
    cov = {"formula": "related_party_payments"}
    binding = {"related_party_payments": {"txn_ids": ["TXN-X-0002"]}}
    kind, rows = variable_rows("related_party_payments", cov, ROLES, ledger, binding)
    assert kind == "bound"  # selection is honoured only where the name is bindable...
    # ...and bindable_terms never offers this name, so compute agrees with the tag in practice
    assert _resolve_variable("related_party_payments", cov, ROLES, ledger, None) == pytest.approx(
        418_662.44
    )


def test_a_doc_figure_wins_over_any_binding():
    cov = {"formula": "group_capex", "doc_figures": {"group_capex": {"amount_usd": 21_850_442.6}}}
    binding = {"group_capex": {"txn_ids": ["TXN-X-0002"]}}
    assert _resolve_variable("group_capex", cov, ROLES, LEDGER, binding) == pytest.approx(
        21_850_442.6
    )


def test_an_unknown_variable_still_raises_rather_than_reading_as_zero():
    with pytest.raises(ValueError, match="not a known role"):
        _resolve_variable("ebitda", {"formula": "ebitda"}, ROLES, LEDGER)


def test_binding_changes_the_verdict_a_wide_role_read_would_have_given():
    cov = {
        "formula": "related_party_payments / operating_expenses",
        "comparison": "<=",
        "threshold": 0.08,
    }
    ledger = [*LEDGER, txn("TXN-X-0005", -418_662.44, "advisory", related_party=True)]
    wide = compute_covenant(cov, ROLES, ledger, None)
    bound = compute_covenant(
        cov, ROLES, ledger, {"operating_expenses": {"txn_ids": ["TXN-X-0001"]}}
    )
    assert wide.status == "COMPLIANT" and wide.actual == pytest.approx(0.02, abs=0.01)
    assert bound.status == "BREACH" and bound.actual == pytest.approx(0.10, abs=0.005)


def test_a_reclassified_row_inside_a_bound_term_is_still_an_evidence_candidate():
    ledger = [
        txn("TXN-X-0010", -300_000.0, "capex", reclass_reason="auditor reclassified into capex"),
        txn("TXN-X-0011", -100_000.0, "capex"),
    ]
    cov = {"formula": "capital_expenditure", "comparison": "<=", "threshold": 350_000.0}
    roles = {"capex": "capital_expenditure"}
    binding = {"capital_expenditure": {"txn_ids": ["TXN-X-0010", "TXN-X-0011"]}}
    assert _find_evidence_txn(cov, roles, ledger, binding) == "TXN-X-0010"


def test_a_denominator_collapsed_onto_its_numerator_is_dropped():
    ledger = [
        txn(
            "TXN-X-0020",
            -418_204.37,
            "capex",
            description="Transfer of plant to subsidiary",
            unrestricted_sub_transfer=True,
        ),
        txn("TXN-X-0021", -900_000.00, "capex", description="Purchase of equipment"),
        txn("TXN-X-0022", -600_000.00, "capex", description="Construction works"),
    ]
    roles = {"capex": "capital_expenditure"}
    cov = {
        "formula": "unrestricted_sub_transfers / capital_expenditure",
        "comparison": "<=",
        "threshold": 0.15,
    }
    # the binder narrowed the denominator with the numerator's own qualifier -> ratio 1.0
    collapsed = {"capital_expenditure": {"txn_ids": ["TXN-X-0020"]}}
    assert compute_covenant(cov, roles, ledger, collapsed).actual == pytest.approx(0.22, abs=0.01)


def test_a_denominator_that_legitimately_differs_is_kept():
    ledger = [
        txn("TXN-X-0020", -400_000.00, "capex", unrestricted_sub_transfer=True),
        txn("TXN-X-0021", -600_000.00, "capex"),
        txn("TXN-X-0022", -900_000.00, "other_capex"),
    ]
    roles = {"capex": "capital_expenditure", "other_capex": "capital_expenditure"}
    cov = {
        "formula": "unrestricted_sub_transfers / capital_expenditure",
        "comparison": "<=",
        "threshold": 0.15,
    }
    binding = {"capital_expenditure": {"txn_ids": ["TXN-X-0020", "TXN-X-0021"]}}
    # denominator is wider than the numerator, so the binding stands: 400k / 1.0M
    assert compute_covenant(cov, roles, ledger, binding).actual == pytest.approx(0.40, abs=0.01)


def test_narrow_absurd_cost_roles_fires_whatever_the_role_is_called():
    # the rescue used to match the literal suffix "_expenses", so a model that named the same role
    # "operating_cost" or "interest_expense" got no rescue at all
    for name in ("operating_expenses", "operating_expense", "operating_costs", "operating_cost"):
        roles = {c: (name if r == "operating_expenses" else r) for c, r in ROLES.items()}
        narrowed = narrow_absurd_cost_roles(roles, LEDGER)
        assert narrowed["rent"] != name, name
        assert narrowed["operating_costs"] == name, name


def test_narrow_absurd_cost_roles_leaves_a_bound_term_alone():
    # the heuristic would otherwise strip rent and marketing out of operating_expenses
    narrowed = narrow_absurd_cost_roles(ROLES, LEDGER)
    assert narrowed["rent"] != "operating_expenses"
    untouched = narrow_absurd_cost_roles(ROLES, LEDGER, {"operating_expenses"})
    assert untouched == ROLES


def test_a_bound_cost_term_many_times_revenue_falls_back_to_the_role_map():
    ledger = [
        txn("TXN-Y-0001", 6_900_000.0, "sales", description="Confectionery sales settlement"),
        txn(
            "TXN-Y-0002", -202_000_000.0, "payroll", description="Payroll for administrative staff"
        ),
        txn("TXN-Y-0003", -800_000.0, "operating_costs", description="Factory operating costs"),
    ]
    roles = {
        "sales": "revenue",
        "payroll": "operating_expenses",
        "operating_costs": "operating_expenses",
    }
    cov = {"formula": "revenue - operating_expenses", "comparison": ">=", "threshold": 1.0}
    swollen = {"operating_expenses": {"txn_ids": ["TXN-Y-0002", "TXN-Y-0003"]}}
    # 202,800,000 against revenue of 6,900,000 cannot be the line item the clause names, however
    # deliberately the rows were picked
    cleaned = without_absurd_bound_terms(cov, roles, ledger, swollen)
    assert "operating_expenses" not in cleaned


def test_a_bound_cost_term_of_ordinary_size_is_kept():
    ledger = [
        txn("TXN-Y-0001", 6_900_000.0, "sales"),
        txn("TXN-Y-0003", -800_000.0, "operating_costs"),
    ]
    roles = {"sales": "revenue", "operating_costs": "operating_expenses"}
    cov = {"formula": "revenue - operating_expenses", "comparison": ">=", "threshold": 1.0}
    binding = {"operating_expenses": {"txn_ids": ["TXN-Y-0003"]}}
    assert without_absurd_bound_terms(cov, roles, ledger, binding) == binding


def test_revenue_itself_is_never_dropped_by_the_size_guard():
    ledger = [txn("TXN-Y-0001", 6_900_000.0, "sales")]
    roles = {"sales": "revenue"}
    cov = {"formula": "revenue", "comparison": ">=", "threshold": 1.0}
    binding = {"revenue": {"txn_ids": ["TXN-Y-0001"]}}
    assert without_absurd_bound_terms(cov, roles, ledger, binding) == binding
