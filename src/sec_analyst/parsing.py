"""Turn a 10-K HTML document into plain text and split it into its standard Items."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

# Standard 10-K items (Form 10-K, General Instructions). Item numbers are unique across Parts.
SECTION_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibit and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

BLOCK_TAGS = [
    "p", "div", "li", "ul", "ol", "table", "tr", "section", "article",
    "h1", "h2", "h3", "h4", "h5", "h6", "title",
]

# A heading line such as "Item 1A. Risk Factors" or "ITEM 7 - MANAGEMENT'S ...".
HEADING_RE = re.compile(r"^item\s+(\d{1,2}[abc]?)\s*(?:[.:\-–—|]|\s|$)\s*(.*)$", re.I)
MIN_SECTION_CHARS = 300  # shorter segments are table-of-contents entries or page headers


@dataclass
class Section:
    section_id: str  # "1A", or "cover" for text before the first Item
    title: str  # "Item 1A. Risk Factors"
    text: str


def section_label(section_id: str) -> str:
    if section_id == "cover":
        return "Cover page"
    return f"Item {section_id}. {SECTION_TITLES.get(section_id, '')}".strip()


def html_to_text(html: str) -> str:
    """Convert filing HTML to text: one paragraph per line, table rows as 'a | b | c'."""
    soup = BeautifulSoup(html, "html.parser")
    # Inline XBRL puts machine-readable facts in a hidden header; drop it and anything hidden.
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.insert_after(" ␟ ")  # temporary cell separator
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    raw = soup.get_text()
    lines = (_clean_line(line) for line in raw.split("\n"))
    return "\n".join(line for line in lines if line)


def _clean_line(line: str) -> str:
    line = line.replace("\xa0", " ").replace("​", "")
    if "␟" not in line:
        return re.sub(r"\s+", " ", line).strip()
    cells = [re.sub(r"\s+", " ", c).strip() for c in line.split("␟")]
    merged: list[str] = []
    pending_prefix = ""
    for cell in cells:
        if not cell:
            continue
        if cell in {"$", "("}:
            pending_prefix += cell
            continue
        if cell in {"%", ")", ")%"} and merged:
            merged[-1] += cell
            continue
        merged.append(pending_prefix + cell)
        pending_prefix = ""
    return " | ".join(merged)


def split_sections(text: str) -> list[Section]:
    """Split filing text into Items. Table-of-contents entries are discarded as too short.

    If an Item heading appears several times (contents page, running headers), all long
    segments for that Item are joined in document order.
    """
    segments: list[tuple[str, list[str]]] = [("cover", [])]
    for line in text.split("\n"):
        item_id = _heading_item(line)
        if item_id:
            segments.append((item_id, []))
        else:
            segments[-1][1].append(line)

    order: list[str] = []
    bodies: dict[str, list[str]] = {}
    for item_id, lines in segments:
        body = "\n".join(lines).strip()
        if len(body) < MIN_SECTION_CHARS:
            continue
        if item_id not in bodies:
            order.append(item_id)
            bodies[item_id] = []
        bodies[item_id].append(body)
    return [Section(i, section_label(i), "\n".join(bodies[i])) for i in order]


def _heading_item(line: str) -> str | None:
    if len(line) > 200:
        return None
    match = HEADING_RE.match(line.strip())
    if not match:
        return None
    item_id = match.group(1).upper()
    rest = match.group(2).lstrip("| ").strip()
    if item_id not in SECTION_TITLES:
        return None
    # "Item 7 of this report ..." is a cross-reference, not a heading.
    if rest and rest[0].islower():
        return None
    return item_id
