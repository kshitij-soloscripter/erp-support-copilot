"""Reject requests that this documentation-only library cannot substantiate."""

import re

ABSTENTION = "Not covered by this library."
OTHER_PRODUCTS = re.compile(r"\b(?:SAP|Oracle|NetSuite|Business\s+Central|Dynamics\s+NAV)\b", re.IGNORECASE)
FINANCIAL_VALUES = re.compile(
    r"\b(?:balance|balances|amount|total|revenue|profit|outstanding|unpaid)\b", re.IGNORECASE
)
PRIVATE_OR_CURRENT = re.compile(
    r"\b(?:our|my|company|today|yesterday|currently|current|last\s+(?:week|month|year))\b", re.IGNORECASE
)
HOW_TO = re.compile(
    r"\b(?:how\s+(?:can|do|does|should|to|is)|where\s+(?:can|do|is)|which\s+(?:report|page|screen))\b",
    re.IGNORECASE,
)


def question_limitation(question: str) -> str | None:
    """Return a precise refusal for known unsupported requests, otherwise None.

    This is a conservative scope guard, not a general answerability classifier.
    Instructional questions about viewing balances remain allowed.
    """
    if OTHER_PRODUCTS.search(question):
        return (
            ABSTENTION
            + " The indexed sources cover Dynamics 365 Finance Accounts payable, not the requested ERP product."
        )
    if (
        FINANCIAL_VALUES.search(question)
        and PRIVATE_OR_CURRENT.search(question)
        and not HOW_TO.search(question)
    ):
        return "I cannot access live company balances or transactions. This library contains public product documentation, not company financial records."
    return None
