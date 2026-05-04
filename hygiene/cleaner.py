"""
hygiene/cleaner.py

CRM email list hygiene toolkit.
Removes undeliverable, low-quality, and disengaged contacts
from a flat CSV export before re-import into a CRM or ESP.

Real-world baseline: 500k → 120k active subscribers (-76%),
resulting in +50% conversion uplift and significant ESP cost reduction.

Usage (CLI):
    python -m hygiene.cleaner --input data/sample/contacts.csv --output data/out/

Usage (library):
    from hygiene.cleaner import load, run_all_checks, save_report

    df = load("contacts.csv", email_col="email")
    clean_df, report = run_all_checks(df, email_col="email")
    save_report(report, "hygiene_report.txt")
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Role / functional addresses that should never receive marketing email
ROLE_PREFIXES = (
    "info", "contact", "support", "help", "admin", "noreply", "no-reply",
    "postmaster", "webmaster", "sales", "billing", "abuse", "security",
    "newsletter", "unsubscribe", "office", "team", "hello", "mail",
)

# Disposable / throwaway domain patterns (non-exhaustive, extend as needed)
DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "tempmail.com", "throwam.com",
    "yopmail.com", "trashmail.com", "sharklasers.com", "dispostable.com",
    "maildrop.cc", "10minutemail.com",
}

# Regex for a valid email address (RFC-5321 simplified)
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load(
    path: str | Path,
    *,
    email_col: str = "email",
    sep: str = ",",
    encoding: str = "ISO-8859-1",
) -> pd.DataFrame:
    """Read a CRM export CSV and normalise the email column."""
    df = pd.read_csv(path, sep=sep, encoding=encoding)
    if email_col not in df.columns:
        raise ValueError(
            f"Column '{email_col}' not found. Available: {list(df.columns)}"
        )
    df[email_col] = df[email_col].astype(str).str.strip().str.lower()
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def save_report(report: dict, path: str | Path) -> None:
    """Write a human-readable hygiene report to a text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Email List Hygiene Report ===\n\n")
        for key, value in report.items():
            f.write(f"{key:<35} {value}\n")
    logger.info("Report saved to %s", path)


# ---------------------------------------------------------------------------
# Individual checks  (each returns a boolean Series: True = remove)
# ---------------------------------------------------------------------------


def flag_invalid_syntax(df: pd.DataFrame, email_col: str) -> pd.Series:
    """Flag addresses that fail basic RFC-5321 syntax validation."""
    return ~df[email_col].str.match(EMAIL_REGEX, na=False)


def flag_duplicates(df: pd.DataFrame, email_col: str) -> pd.Series:
    """Flag all but the first occurrence of duplicate email addresses."""
    return df[email_col].duplicated(keep="first")


def flag_role_addresses(df: pd.DataFrame, email_col: str) -> pd.Series:
    """Flag role / functional addresses (info@, support@, noreply@, …)."""
    pattern = "^(" + "|".join(re.escape(p) for p in ROLE_PREFIXES) + r")[@+\-_\.]"
    return df[email_col].str.match(pattern, na=False)


def flag_disposable_domains(df: pd.DataFrame, email_col: str) -> pd.Series:
    """Flag addresses from known disposable / throwaway email providers."""
    domains = df[email_col].str.split("@").str[-1]
    return domains.isin(DISPOSABLE_DOMAINS)


def flag_engagement_decay(
    df: pd.DataFrame,
    email_col: str,
    *,
    last_open_col: str | None = None,
    last_click_col: str | None = None,
    no_open_days: int = 180,
    no_click_days: int = 365,
) -> pd.Series:
    """
    Flag contacts with no opens in *no_open_days* AND no clicks in
    *no_click_days*. If neither date column is present, returns all-False
    (no contacts flagged) so the function is safe to call without engagement
    data.
    """
    flag = pd.Series(False, index=df.index)

    has_open  = last_open_col  and last_open_col  in df.columns
    has_click = last_click_col and last_click_col in df.columns

    if not has_open and not has_click:
        logger.warning(
            "No engagement columns provided — skipping engagement decay check."
        )
        return flag

    today = pd.Timestamp.today().normalize()

    if has_open:
        last_open = pd.to_datetime(df[last_open_col], errors="coerce")
        open_days_ago = (today - last_open).dt.days
        stale_open = open_days_ago.isna() | (open_days_ago > no_open_days)
    else:
        stale_open = pd.Series(True, index=df.index)

    if has_click:
        last_click = pd.to_datetime(df[last_click_col], errors="coerce")
        click_days_ago = (today - last_click).dt.days
        stale_click = click_days_ago.isna() | (click_days_ago > no_click_days)
    else:
        stale_click = pd.Series(True, index=df.index)

    # Only flag if BOTH open and click are stale (conservative approach)
    return stale_open & stale_click


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_checks(
    df: pd.DataFrame,
    email_col: str = "email",
    *,
    last_open_col: str | None = None,
    last_click_col: str | None = None,
    no_open_days: int = 180,
    no_click_days: int = 365,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all hygiene checks and return (clean_df, report_dict).

    The report_dict contains row counts for each removal reason and
    overall before/after totals.
    """
    total_before = len(df)
    removal_flags: dict[str, pd.Series] = {}

    removal_flags["invalid_syntax"]    = flag_invalid_syntax(df, email_col)
    removal_flags["duplicates"]        = flag_duplicates(df, email_col)
    removal_flags["role_addresses"]    = flag_role_addresses(df, email_col)
    removal_flags["disposable_domain"] = flag_disposable_domains(df, email_col)
    removal_flags["engagement_decay"]  = flag_engagement_decay(
        df, email_col,
        last_open_col=last_open_col,
        last_click_col=last_click_col,
        no_open_days=no_open_days,
        no_click_days=no_click_days,
    )

    # Combined mask: remove if flagged by ANY check
    remove_mask = pd.concat(removal_flags.values(), axis=1).any(axis=1)
    clean_df = df[~remove_mask].copy()

    total_after   = len(clean_df)
    total_removed = total_before - total_after

    report = {
        "contacts_before":       total_before,
        "contacts_after":        total_after,
        "total_removed":         total_removed,
        "removal_rate_%":        round(total_removed / max(total_before, 1) * 100, 1),
    }
    for reason, flags in removal_flags.items():
        report[f"removed_{reason}"] = int(flags.sum())

    logger.info(
        "Hygiene complete: %d → %d contacts (-%d, %.1f%%)",
        total_before, total_after, total_removed, report["removal_rate_%"],
    )
    return clean_df, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run email list hygiene checks on a CRM CSV export."
    )
    p.add_argument("--input",          required=True,  help="Path to input CSV.")
    p.add_argument("--output",         default="data/out/", help="Output directory.")
    p.add_argument("--email-col",      default="email", help="Email column name.")
    p.add_argument("--last-open-col",  default=None,   help="Last open date column.")
    p.add_argument("--last-click-col", default=None,   help="Last click date column.")
    p.add_argument("--no-open-days",   type=int, default=180)
    p.add_argument("--no-click-days",  type=int, default=365)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    df = load(args.input, email_col=args.email_col)

    clean_df, report = run_all_checks(
        df,
        email_col=args.email_col,
        last_open_col=args.last_open_col,
        last_click_col=args.last_click_col,
        no_open_days=args.no_open_days,
        no_click_days=args.no_click_days,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    clean_path = out / (Path(args.input).stem + "_clean.csv")
    clean_df.to_csv(clean_path, index=False, encoding="ISO-8859-1")
    logger.info("Clean list saved: %s", clean_path)

    save_report(report, out / "hygiene_report.txt")

    print("\n=== Summary ===")
    for k, v in report.items():
        print(f"  {k:<35} {v}")


if __name__ == "__main__":
    main()
