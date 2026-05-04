"""
tests/test_cleaner.py

Unit tests for hygiene.cleaner.
Run with:  pytest tests/ -v
"""

import pandas as pd
import pytest

from hygiene.cleaner import (
    flag_invalid_syntax,
    flag_duplicates,
    flag_role_addresses,
    flag_disposable_domains,
    flag_engagement_decay,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(*emails: str) -> pd.DataFrame:
    return pd.DataFrame({"email": list(emails)})


# ---------------------------------------------------------------------------
# flag_invalid_syntax
# ---------------------------------------------------------------------------

class TestInvalidSyntax:
    def test_valid_emails_not_flagged(self):
        df = make_df("user@example.com", "name.surname@company.org")
        assert flag_invalid_syntax(df, "email").sum() == 0

    def test_missing_at_flagged(self):
        df = make_df("notanemail", "also-bad")
        assert flag_invalid_syntax(df, "email").sum() == 2

    def test_missing_domain_flagged(self):
        df = make_df("user@", "user@.")
        assert flag_invalid_syntax(df, "email").sum() == 2

    def test_mixed(self):
        df = make_df("good@example.com", "bad-email", "also@good.de")
        result = flag_invalid_syntax(df, "email")
        assert result.tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# flag_duplicates
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_no_duplicates(self):
        df = make_df("a@x.com", "b@x.com", "c@x.com")
        assert flag_duplicates(df, "email").sum() == 0

    def test_one_duplicate_flagged(self):
        df = make_df("a@x.com", "b@x.com", "a@x.com")
        result = flag_duplicates(df, "email")
        assert result.tolist() == [False, False, True]

    def test_first_occurrence_kept(self):
        df = make_df("a@x.com", "a@x.com", "a@x.com")
        result = flag_duplicates(df, "email")
        assert result.tolist() == [False, True, True]


# ---------------------------------------------------------------------------
# flag_role_addresses
# ---------------------------------------------------------------------------

class TestRoleAddresses:
    def test_info_flagged(self):
        df = make_df("info@company.com")
        assert flag_role_addresses(df, "email").sum() == 1

    def test_noreply_flagged(self):
        df = make_df("noreply@company.com", "no-reply@company.com")
        assert flag_role_addresses(df, "email").sum() == 2

    def test_real_user_not_flagged(self):
        df = make_df("john.doe@company.com", "maria@example.org")
        assert flag_role_addresses(df, "email").sum() == 0

    def test_support_and_admin_flagged(self):
        df = make_df("support@x.com", "admin@x.com", "user@x.com")
        result = flag_role_addresses(df, "email")
        assert result.tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# flag_disposable_domains
# ---------------------------------------------------------------------------

class TestDisposableDomains:
    def test_known_disposable_flagged(self):
        df = make_df("user@mailinator.com", "test@yopmail.com")
        assert flag_disposable_domains(df, "email").sum() == 2

    def test_real_domain_not_flagged(self):
        df = make_df("user@gmail.com", "user@company.de")
        assert flag_disposable_domains(df, "email").sum() == 0


# ---------------------------------------------------------------------------
# flag_engagement_decay
# ---------------------------------------------------------------------------

class TestEngagementDecay:
    def _make_engagement_df(self, last_open: str, last_click: str) -> pd.DataFrame:
        return pd.DataFrame({
            "email":      ["user@example.com"],
            "last_open":  [last_open],
            "last_click": [last_click],
        })

    def test_recent_open_not_flagged(self):
        df = self._make_engagement_df("2025-04-01", "2024-06-01")
        result = flag_engagement_decay(
            df, "email",
            last_open_col="last_open",
            last_click_col="last_click",
        )
        assert result.sum() == 0

    def test_stale_open_and_click_flagged(self):
        df = self._make_engagement_df("2020-01-01", "2020-01-01")
        result = flag_engagement_decay(
            df, "email",
            last_open_col="last_open",
            last_click_col="last_click",
        )
        assert result.sum() == 1

    def test_stale_open_but_recent_click_not_flagged(self):
        df = self._make_engagement_df("2020-01-01", "2025-04-01")
        result = flag_engagement_decay(
            df, "email",
            last_open_col="last_open",
            last_click_col="last_click",
        )
        assert result.sum() == 0

    def test_no_engagement_cols_returns_all_false(self):
        df = make_df("user@example.com")
        result = flag_engagement_decay(df, "email")
        assert result.sum() == 0

    def test_missing_date_treated_as_stale(self):
        df = pd.DataFrame({
            "email":      ["user@example.com"],
            "last_open":  [None],
            "last_click": [None],
        })
        result = flag_engagement_decay(
            df, "email",
            last_open_col="last_open",
            last_click_col="last_click",
        )
        assert result.sum() == 1


# ---------------------------------------------------------------------------
# run_all_checks (integration)
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_clean_list_unchanged(self):
        df = make_df("alice@example.com", "bob@example.com")
        clean, report = run_all_checks(df)
        assert len(clean) == 2
        assert report["total_removed"] == 0

    def test_removes_invalid_syntax(self):
        df = make_df("good@example.com", "notanemail")
        clean, report = run_all_checks(df)
        assert len(clean) == 1
        assert report["removed_invalid_syntax"] == 1

    def test_removes_role_address(self):
        df = make_df("info@company.com", "user@company.com")
        clean, report = run_all_checks(df)
        assert len(clean) == 1
        assert report["removed_role_addresses"] == 1

    def test_report_contains_all_keys(self):
        df = make_df("user@example.com")
        _, report = run_all_checks(df)
        expected_keys = {
            "contacts_before", "contacts_after", "total_removed",
            "removal_rate_%", "removed_invalid_syntax", "removed_duplicates",
            "removed_role_addresses", "removed_disposable_domain",
            "removed_engagement_decay",
        }
        assert expected_keys.issubset(report.keys())

    def test_removal_rate_calculation(self):
        df = make_df("good@example.com", "bad-email", "info@x.com", "other@x.com")
        _, report = run_all_checks(df)
        expected_rate = round(report["total_removed"] / 4 * 100, 1)
        assert report["removal_rate_%"] == expected_rate
