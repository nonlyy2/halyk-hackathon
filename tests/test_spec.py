from __future__ import annotations

from covenant.analyze.spec import (
    _coerce_numbers,
    _drop_double_counted_carve_outs,
    _extract_names,
    basket_not_subtracted,
    self_reported_approximations,
    validate_spec,
)


def test_a_threshold_written_as_text_is_read_back_as_a_number():
    spec = {"covenants": {"6.1": {"threshold": "0.08"}, "6.2": {"threshold": "$3,000,000.00"}}}
    coerced = _coerce_numbers(spec)["covenants"]
    assert coerced["6.1"]["threshold"] == 0.08
    assert coerced["6.2"]["threshold"] == 3_000_000.0


def test_a_ratio_keeps_its_value_without_the_x_suffix():
    assert (
        _coerce_numbers({"covenants": {"6.1": {"threshold": "1.20x"}}})["covenants"]["6.1"][
            "threshold"
        ]
        == 1.2
    )


def test_carve_out_allowances_and_doc_figures_are_coerced_too():
    spec = {
        "covenants": {
            "6.1": {
                "carve_out": {"kind": "allowance", "allowance": "$200,000.00"},
                "doc_figures": {"group_capex": {"amount_usd": "21,850,442.60"}},
            }
        }
    }
    cov = _coerce_numbers(spec)["covenants"]["6.1"]
    assert cov["carve_out"]["allowance"] == 200_000.0
    assert cov["doc_figures"]["group_capex"]["amount_usd"] == 21_850_442.60


def test_a_threshold_that_is_not_a_number_is_left_alone():
    spec = {"covenants": {"6.1": {"threshold": "as agreed with the Lender"}}}
    assert _coerce_numbers(spec)["covenants"]["6.1"]["threshold"] == "as agreed with the Lender"


def test_a_numeric_threshold_is_untouched():
    assert (
        _coerce_numbers({"covenants": {"6.1": {"threshold": 1.7}}})["covenants"]["6.1"]["threshold"]
        == 1.7
    )


def test_validate_spec_names_a_variable_that_resolves_to_nothing():
    spec = {"roles": {"sales": "revenue"}, "covenants": {"6.1": {"formula": "ebitda - revenue"}}}
    assert validate_spec(spec) == {"6.1": ["ebitda"]}


def test_validate_spec_reports_an_unparseable_formula_rather_than_raising():
    spec = {"roles": {}, "covenants": {"6.1": {"formula": "revenue - (operating"}}}
    assert "unparseable" in validate_spec(spec)["6.1"][0]


def test_extract_names_ignores_the_allowed_function_names():
    assert _extract_names("min(payroll, utilities)") == {"payroll", "utilities"}


def test_a_formula_the_model_calls_a_proxy_is_flagged():
    spec = {
        "covenants": {
            "6.1": {
                "formula": "capital_expenditure",
                "metric": "Total value of assets transferred to unrestricted subsidiaries",
                "notes": ["capital_expenditure is used as a proxy for such transfers."],
            }
        }
    }
    assert "6.1" in self_reported_approximations(spec)


def test_a_condition_the_model_says_it_cannot_evaluate_is_flagged():
    spec = {
        "covenants": {
            "6.1": {
                "formula": "related_party_payments",
                "metric": "",
                "notes": [
                    "Applies only when leverage exceeds 3.00x; condition cannot be evaluated."
                ],
            }
        }
    }
    assert "6.1" in self_reported_approximations(spec)


def test_a_term_the_model_says_is_missing_from_the_data_is_flagged():
    spec = {
        "covenants": {
            "6.2": {
                "formula": "revenue - operating_expenses",
                "metric": "Liquidity contribution",
                "notes": ["Financing receipts are not represented in the available data."],
            }
        }
    }
    assert "6.2" in self_reported_approximations(spec)


def test_an_ordinary_note_is_not_flagged():
    spec = {
        "covenants": {
            "6.1": {
                "formula": "revenue - operating_expenses",
                "metric": "EBITDA for the period",
                "notes": ["The clause defines EBITDA as revenue less operating expenses."],
            }
        }
    }
    assert self_reported_approximations(spec) == {}


def test_a_carve_out_repeating_what_the_formula_already_caps_is_dropped():
    # "after excluding a permitted basket of up to $500,000" belongs in the formula once. Recorded
    # also as an exclusion it is applied twice, and the second time uncapped -- the borrower gets
    # an unlimited basket and the verdict flips while `actual` stays right.
    spec = {
        "covenants": {
            "6.3": {
                "formula": "related_party_payments - min(consulting_services, 500000)",
                "carve_out": {"kind": "exclusion", "excluded_role": "consulting_services"},
            }
        }
    }
    assert _drop_double_counted_carve_outs(spec)["covenants"]["6.3"]["carve_out"] is None


def test_a_genuine_exclusion_of_something_the_formula_does_not_cap_is_kept():
    carve = {"kind": "exclusion", "excluded_role": "intra_group_transfers"}
    spec = {"covenants": {"6.1": {"formula": "operating_expenses", "carve_out": dict(carve)}}}
    assert _drop_double_counted_carve_outs(spec)["covenants"]["6.1"]["carve_out"] == carve


def test_an_allowance_carve_out_is_untouched():
    carve = {"kind": "allowance", "allowance": 200000}
    spec = {"covenants": {"6.1": {"formula": "min(a, 5)", "carve_out": dict(carve)}}}
    assert _drop_double_counted_carve_outs(spec)["covenants"]["6.1"]["carve_out"] == carve


def test_a_clause_granting_a_basket_with_no_subtraction_is_flagged():
    spec = {
        "clauses": {
            "6.1": "не превышать $250,000.00, после исключения разрешённой корзины до $300,000.00"
        },
        "covenants": {"6.1": {"formula": "related_party_payments"}},
    }
    assert basket_not_subtracted(spec) == ["6.1"]


def test_a_basket_clause_whose_formula_subtracts_is_not_flagged():
    spec = {
        "clauses": {
            "6.1": "не превышать $250,000.00, после исключения разрешённой корзины до $300,000.00"
        },
        "covenants": {
            "6.1": {"formula": "related_party_payments - min(consulting_services, 300000)"}
        },
    }
    assert basket_not_subtracted(spec) == []
