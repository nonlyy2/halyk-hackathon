from __future__ import annotations

from covenant.analyze.spec import _coerce_numbers, _extract_names, validate_spec


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
