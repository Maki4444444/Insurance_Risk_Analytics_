import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_claim_frequency(group: pd.DataFrame) -> float:
    """
    Claim Frequency = proportion of policies with at least one claim.
    """
    if group.empty:
        logger.warning("Empty group passed to get_claim_frequency.")
        return 0.0
    return (group["TotalClaims"] > 0).mean()


def get_claim_severity(group: pd.DataFrame) -> pd.Series:
    """
    Claim Severity = claim amounts where a claim occurred.
    """
    if group.empty:
        logger.warning("Empty group passed to get_claim_severity.")
        return pd.Series(dtype=float)
    return group[group["TotalClaims"] > 0]["TotalClaims"]


def get_margin(group: pd.DataFrame) -> pd.Series:
    """
    Margin = TotalPremium - TotalClaims per policy.
    """
    if group.empty:
        logger.warning("Empty group passed to get_margin.")
        return pd.Series(dtype=float)
    return group["TotalPremium"] - group["TotalClaims"]


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """
    Cohen's d effect size for two groups.
    Small=0.2, Medium=0.5, Large=0.8
    """
    if group_a.empty or group_b.empty:
        logger.warning("Empty series passed to cohens_d.")
        return 0.0
    pooled_std = np.sqrt((group_a.std() ** 2 + group_b.std() ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return (group_a.mean() - group_b.mean()) / pooled_std


def effect_size_label(d: float) -> str:
    """
    Return a human-readable effect size label.
    """
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    return "Large"


# ── Statistical Tests ─────────────────────────────────────────────────────────
def chi_squared_test(group_a: pd.DataFrame, group_b: pd.DataFrame,
                     label_a: str = "Group A", label_b: str = "Group B") -> dict:
    """
    Chi-squared test for claim frequency (categorical KPI).
    """
    try:
        if group_a.empty or group_b.empty:
            raise ValueError(f"One or both groups are empty: {label_a}={len(group_a)}, "
                             f"{label_b}={len(group_b)}")

        claimed_a = (group_a["TotalClaims"] > 0).sum()
        not_claimed_a = (group_a["TotalClaims"] == 0).sum()
        claimed_b = (group_b["TotalClaims"] > 0).sum()
        not_claimed_b = (group_b["TotalClaims"] == 0).sum()

        if claimed_a == 0 and claimed_b == 0:
            raise ValueError("Both groups have zero claims — chi-squared test not applicable.")

        contingency_table = [[claimed_a, not_claimed_a],
                             [claimed_b, not_claimed_b]]

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        freq_a = get_claim_frequency(group_a) * 100
        freq_b = get_claim_frequency(group_b) * 100

        return {
            "test": "Chi-Squared",
            "statistic": round(chi2, 4),
            "p_value": round(p_value, 6),
            "dof": dof,
            "claim_rate_a": round(freq_a, 2),
            "claim_rate_b": round(freq_b, 2),
            "label_a": label_a,
            "label_b": label_b,
            "decision": "Reject H0" if p_value < 0.05 else "Fail to Reject H0"
        }

    except ValueError as e:
        logger.error(f"Chi-squared test failed: {e}")
        return {
            "test": "Chi-Squared",
            "statistic": None,
            "p_value": None,
            "dof": None,
            "claim_rate_a": None,
            "claim_rate_b": None,
            "label_a": label_a,
            "label_b": label_b,
            "decision": f"Test failed: {e}"
        }
    except Exception as e:
        logger.error(f"Unexpected error in chi_squared_test: {e}")
        return {
            "test": "Chi-Squared",
            "statistic": None,
            "p_value": None,
            "dof": None,
            "claim_rate_a": None,
            "claim_rate_b": None,
            "label_a": label_a,
            "label_b": label_b,
            "decision": f"Test failed: {e}"
        }


def t_test(group_a: pd.Series, group_b: pd.Series,
           label_a: str = "Group A", label_b: str = "Group B") -> dict:
    """
    Two-sample Welch's t-test for numerical KPIs.
    """
    try:
        a = group_a.dropna()
        b = group_b.dropna()

        if a.empty:
            raise ValueError(f"Group A ({label_a}) has no valid values after "
                             f"dropping nulls — t-test not applicable.")
        if b.empty:
            raise ValueError(f"Group B ({label_b}) has no valid values after "
                             f"dropping nulls — t-test not applicable.")
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"Insufficient sample size: "
                             f"{label_a}={len(a)}, {label_b}={len(b)}. "
                             f"Minimum 2 observations required per group.")

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)

        return {
            "test": "T-Test (Welch)",
            "statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "mean_a": round(a.mean(), 2),
            "mean_b": round(b.mean(), 2),
            "mean_diff": round(a.mean() - b.mean(), 2),
            "cohens_d": round(d, 3),
            "effect_size": effect_size_label(d),
            "label_a": label_a,
            "label_b": label_b,
            "decision": "Reject H0" if p_value < 0.05 else "Fail to Reject H0"
        }

    except ValueError as e:
        logger.error(f"T-test failed: {e}")
        return {
            "test": "T-Test (Welch)",
            "statistic": None,
            "p_value": None,
            "mean_a": None,
            "mean_b": None,
            "mean_diff": None,
            "cohens_d": None,
            "effect_size": None,
            "label_a": label_a,
            "label_b": label_b,
            "decision": f"Test failed: {e}"
        }
    except Exception as e:
        logger.error(f"Unexpected error in t_test: {e}")
        return {
            "test": "T-Test (Welch)",
            "statistic": None,
            "p_value": None,
            "mean_a": None,
            "mean_b": None,
            "mean_diff": None,
            "cohens_d": None,
            "effect_size": None,
            "label_a": label_a,
            "label_b": label_b,
            "decision": f"Test failed: {e}"
        }


# ── Hypothesis Tests ──────────────────────────────────────────────────────────
def test_province_risk(df: pd.DataFrame,
                       province_a: str, province_b: str) -> pd.DataFrame:
    """
    H1: There are no risk differences across provinces.
    """
    try:
        if "Province" not in df.columns:
            raise KeyError("Column 'Province' not found in DataFrame.")

        group_a = df[df["Province"] == province_a]
        group_b = df[df["Province"] == province_b]

        if group_a.empty:
            raise ValueError(f"Province '{province_a}' not found in data.")
        if group_b.empty:
            raise ValueError(f"Province '{province_b}' not found in data.")

        freq = chi_squared_test(group_a, group_b, province_a, province_b)
        sev = t_test(get_claim_severity(group_a), get_claim_severity(group_b),
                     province_a, province_b)

        return pd.DataFrame([
            {
                "Hypothesis": "H1: No risk differences across provinces",
                "Group A": province_a,
                "Group B": province_b,
                "KPI": "Claim Frequency",
                "Test": freq["test"],
                "Statistic": freq["statistic"],
                "P-Value": freq["p_value"],
                "Decision": freq["decision"]
            },
            {
                "Hypothesis": "H1: No risk differences across provinces",
                "Group A": province_a,
                "Group B": province_b,
                "KPI": "Claim Severity",
                "Test": sev["test"],
                "Statistic": sev["statistic"],
                "P-Value": sev["p_value"],
                "Decision": sev["decision"]
            }
        ])

    except (KeyError, ValueError) as e:
        logger.error(f"test_province_risk failed: {e}")
        return pd.DataFrame([{"Hypothesis": "H1", "Error": str(e)}])
    except Exception as e:
        logger.error(f"Unexpected error in test_province_risk: {e}")
        return pd.DataFrame([{"Hypothesis": "H1", "Error": str(e)}])


def test_postalcode_risk(df: pd.DataFrame,
                         code_a, code_b) -> pd.DataFrame:
    """
    H2: There are no risk differences between zip codes.
    """
    try:
        if "PostalCode" not in df.columns:
            raise KeyError("Column 'PostalCode' not found in DataFrame.")

        group_a = df[df["PostalCode"] == code_a]
        group_b = df[df["PostalCode"] == code_b]

        if group_a.empty:
            raise ValueError(f"PostalCode '{code_a}' not found in data.")
        if group_b.empty:
            raise ValueError(f"PostalCode '{code_b}' not found in data.")

        freq = chi_squared_test(group_a, group_b, str(code_a), str(code_b))
        sev = t_test(get_claim_severity(group_a), get_claim_severity(group_b),
                     str(code_a), str(code_b))

        return pd.DataFrame([
            {
                "Hypothesis": "H2: No risk differences between zip codes",
                "Group A": str(code_a),
                "Group B": str(code_b),
                "KPI": "Claim Frequency",
                "Test": freq["test"],
                "Statistic": freq["statistic"],
                "P-Value": freq["p_value"],
                "Decision": freq["decision"]
            },
            {
                "Hypothesis": "H2: No risk differences between zip codes",
                "Group A": str(code_a),
                "Group B": str(code_b),
                "KPI": "Claim Severity",
                "Test": sev["test"],
                "Statistic": sev["statistic"],
                "P-Value": sev["p_value"],
                "Decision": sev["decision"]
            }
        ])

    except (KeyError, ValueError) as e:
        logger.error(f"test_postalcode_risk failed: {e}")
        return pd.DataFrame([{"Hypothesis": "H2", "Error": str(e)}])
    except Exception as e:
        logger.error(f"Unexpected error in test_postalcode_risk: {e}")
        return pd.DataFrame([{"Hypothesis": "H2", "Error": str(e)}])


def test_postalcode_margin(df: pd.DataFrame,
                           code_a, code_b) -> pd.DataFrame:
    """
    H3: There is no significant margin difference between zip codes.
    """
    try:
        if "PostalCode" not in df.columns:
            raise KeyError("Column 'PostalCode' not found in DataFrame.")

        group_a = df[df["PostalCode"] == code_a]
        group_b = df[df["PostalCode"] == code_b]

        if group_a.empty:
            raise ValueError(f"PostalCode '{code_a}' not found in data.")
        if group_b.empty:
            raise ValueError(f"PostalCode '{code_b}' not found in data.")

        margin = t_test(get_margin(group_a), get_margin(group_b),
                        str(code_a), str(code_b))

        return pd.DataFrame([
            {
                "Hypothesis": "H3: No margin difference between zip codes",
                "Group A": str(code_a),
                "Group B": str(code_b),
                "KPI": "Margin",
                "Test": margin["test"],
                "Statistic": margin["statistic"],
                "P-Value": margin["p_value"],
                "Decision": margin["decision"]
            }
        ])

    except (KeyError, ValueError) as e:
        logger.error(f"test_postalcode_margin failed: {e}")
        return pd.DataFrame([{"Hypothesis": "H3", "Error": str(e)}])
    except Exception as e:
        logger.error(f"Unexpected error in test_postalcode_margin: {e}")
        return pd.DataFrame([{"Hypothesis": "H3", "Error": str(e)}])


def test_gender_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    H4: There is no significant risk difference between Women and Men.
    Filters out 'Not specified' for a clean comparison.
    """
    try:
        if "Gender" not in df.columns:
            raise KeyError("Column 'Gender' not found in DataFrame.")

        gender_df = df[df["Gender"].isin(["Female", "Male"])].copy()

        if gender_df.empty:
            raise ValueError("No Male or Female records found in data.")

        group_a = gender_df[gender_df["Gender"] == "Female"]
        group_b = gender_df[gender_df["Gender"] == "Male"]

        if group_a.empty:
            raise ValueError("No Female records found after filtering.")
        if group_b.empty:
            raise ValueError("No Male records found after filtering.")

        freq = chi_squared_test(group_a, group_b, "Female", "Male")
        sev = t_test(get_claim_severity(group_a), get_claim_severity(group_b),
                     "Female", "Male")

        return pd.DataFrame([
            {
                "Hypothesis": "H4: No risk difference between Women and Men",
                "Group A": "Female",
                "Group B": "Male",
                "KPI": "Claim Frequency",
                "Test": freq["test"],
                "Statistic": freq["statistic"],
                "P-Value": freq["p_value"],
                "Decision": freq["decision"]
            },
            {
                "Hypothesis": "H4: No risk difference between Women and Men",
                "Group A": "Female",
                "Group B": "Male",
                "KPI": "Claim Severity",
                "Test": sev["test"],
                "Statistic": sev["statistic"],
                "P-Value": sev["p_value"],
                "Decision": sev["decision"]
            }
        ])

    except (KeyError, ValueError) as e:
        logger.error(f"test_gender_risk failed: {e}")
        return pd.DataFrame([{"Hypothesis": "H4", "Error": str(e)}])
    except Exception as e:
        logger.error(f"Unexpected error in test_gender_risk: {e}")
        return pd.DataFrame([{"Hypothesis": "H4", "Error": str(e)}])


def build_results_table(all_results: list) -> pd.DataFrame:
    """
    Combine all hypothesis test results into a single summary table.
    """
    try:
        if not all_results:
            raise ValueError("No results provided to build_results_table.")
        return pd.concat(all_results, ignore_index=True)
    except Exception as e:
        logger.error(f"build_results_table failed: {e}")
        return pd.DataFrame()