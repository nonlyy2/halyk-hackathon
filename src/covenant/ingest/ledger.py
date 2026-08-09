import re
from dataclasses import dataclass, field

import pandas as pd

# The case states the rule plainly: a transaction id begins with its borrower's scenario id. It
# says nothing about what follows, and one borrower numbers its rows TXN-KC-CAP-29 rather than
# TXN-KC-0029. Requiring a bare number to the end of the string dropped that borrower's account
# entirely -- no account meant no documents matched, which meant no agreement, no spec and three
# cells lost, none of it reported as an error. Take the segment the rule actually names and leave
# the rest alone.
TXN_ID_RE = re.compile(r"^TXN-([A-Za-z0-9]+)-")


def _scenario_id_from_txn_id(txn_id: str) -> str | None:
    match = TXN_ID_RE.match(txn_id)
    return match.group(1) if match else None


@dataclass(frozen=True)
class Ledger:
    df: pd.DataFrame
    account_to_scenario: dict[str, str] = field(default_factory=dict)
    scenario_to_accounts: dict[str, list[str]] = field(default_factory=dict)

    def scenario_ids(self) -> list[str]:
        return sorted(self.scenario_to_accounts)

    def transactions_for(self, scenario_id: str) -> pd.DataFrame:
        return self.df[self.df["scenario_id"] == scenario_id]


def load_ledger(csv_path: str) -> Ledger:
    df = pd.read_csv(csv_path)
    df["scenario_id"] = df["txn_id"].map(_scenario_id_from_txn_id)

    account_to_scenario = (
        df.dropna(subset=["scenario_id"])
        .drop_duplicates("account_id")
        .set_index("account_id")["scenario_id"]
        .to_dict()
    )
    scenario_to_accounts: dict[str, list[str]] = {}
    for account_id, scenario_id in account_to_scenario.items():
        scenario_to_accounts.setdefault(scenario_id, []).append(account_id)

    return Ledger(
        df=df, account_to_scenario=account_to_scenario, scenario_to_accounts=scenario_to_accounts
    )
