import re

def _roman_to_int(s: str) -> int | None:
    """Convert Roman numeral string to int. Returns None if not a valid Roman numeral."""
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    s = s.upper().strip()
    if not s or not all(c in roman for c in s):
        return None
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and roman[s[i]] < roman[s[i + 1]]:
            result -= roman[s[i]]
        else:
            result += roman[s[i]]
    return result


def _normalize_label(raw_label: str) -> str:
    """
    Normalize any figure/table label to a consistent format.
    Examples:
        "TABLE IV."  → "Table 4"
        "TABLE 4."   → "Table 4"
        "Fig. 3."    → "Figure 3"
        "Fig 4"      → "Figure 4"
        "Figure 4:"  → "Figure 4"
    """
    raw = raw_label.strip().rstrip('.:')

    # ALL-CAPS TABLE + Roman numeral: "TABLE IV"
    m = re.match(r'TABLE\s+([IVXLCDM]+)$', raw, re.IGNORECASE)
    if m:
        num = _roman_to_int(m.group(1))
        if num:
            return f"Table {num}"

    # TABLE + arabic: "TABLE 4"
    m = re.match(r'TABLE\s+(\d+[a-z]?)$', raw, re.IGNORECASE)
    if m:
        return f"Table {m.group(1)}"

    # Fig./Fig → Figure
    raw = re.sub(r'\bFig\.?\b', 'Figure', raw, flags=re.IGNORECASE)

    # Strip trailing period/colon from number: "Figure 4." → "Figure 4"
    raw = re.sub(
        r'((?:Figure|Table)\s+\d+[a-z]?)[.\s]*$', r'\1', raw, flags=re.IGNORECASE
    )

    return raw.strip()


# Reference phrases that indicate a citation to a figure, NOT a caption
_REFERENCE_PATTERNS = re.compile(
    r'as (?:we |you )?(?:can )?see in'
    r'|(?:is|as) (?:shown|illustrated|depicted|presented) in'
    r'|(?:see|refer to)\s*$'
    r'|is (?:a good|an? example|a visual)'
    r'|(?:in|from|using|according to)\s+(?:fig(?:ure)?|table)\s*\d'
    r'|(?:above|below|following|previous)\s+(?:fig(?:ure)?|table)',
    re.IGNORECASE
)


def build_caption_map_from_markdown(markdown_text: str) -> dict[int, list[dict]]:
    """
    Extract figure/table captions from LlamaParse markdown by page.

    Two passes per page:
      Pass 1 — formal standalone caption lines
               e.g. "Fig. 3. Description", "TABLE IV. Title", "Figure 4 Example"
      Pass 2 — inline mentions with no formal caption (fallback)
               e.g. "...as seen in Figure 1, the trend..."
               Skipped if the sentence reads like a reference, not a caption.

    Returns:
        {page_num: [{"label": "Figure 4", "full_caption": "...",
                     "normalized": "figure4", "confidence": "high"|"low"}]}
    """
    pages = re.split(r'---\s*Page\s*(\d+)\s*---', markdown_text)
    caption_map: dict[int, list[dict]] = {}

    for i in range(1, len(pages), 2):
        page_num = int(pages[i])
        page_text = pages[i + 1] if i + 1 < len(pages) else ""

        # Strip markdown emphasis so "**Fig. 3.**" matches cleanly
        clean_text = re.sub(r'\*+', '', page_text)
        clean_text = re.sub(r'_{1,2}', '', clean_text)

        found_labels: set[str] = set()
        page_captions: list[dict] = []

        # ── Pass 1: formal caption lines ───────────────────────────────────
        formal_matches = re.findall(
            r'('
            # Fig/Figure + number + separator + description
            r'(?:Fig(?:ure)?\.?\s*\d+[a-z]?(?:[.:\s]\s*|\s+)[^\n]{3,200})'
            r'|'
            # ALL-CAPS TABLE + Roman or arabic + description
            r'(?:TABLE\s+(?:[IVXLCDM]+|\d+)[a-z]?\.?\s+[^\n]{3,200})'
            r'|'
            # Title-case Table + arabic + separator + description
            r'(?:Table\s+\d+[a-z]?[.:\s]\s*[^\n]{3,200})'
            r')',
            clean_text,
            re.IGNORECASE
        )

        for cap in formal_matches:
            cap = cap.strip()

            # Extract raw label from the start of the caption
            label_match = re.match(
                r'((?:Fig(?:ure)?\.?\s*\d+[a-z]?)'
                r'|(?:TABLE\s+(?:[IVXLCDM]+|\d+)[a-z]?)'
                r'|(?:Table\s+\d+[a-z]?))',
                cap, re.IGNORECASE
            )
            if not label_match:
                continue

            label = _normalize_label(label_match.group(1))
            normalized = re.sub(r'[.\s]', '', label.lower())  # "figure4", "table3"

            if normalized not in found_labels:
                found_labels.add(normalized)
                page_captions.append({
                    "label": label,
                    "full_caption": cap[:200],
                    "normalized": normalized,
                    "confidence": "high",
                })

        # ── Pass 2: inline mentions without a formal caption ───────────────
        # Only used when no formal caption exists for this figure number.
        inline_refs = re.findall(
            r'\b(Fig(?:ure)?\.?\s*\d+[a-z]?)\b',
            clean_text,
            re.IGNORECASE
        )

        for ref in inline_refs:
            ref_clean = ref.strip()
            # Normalize to check deduplication
            num_match = re.search(r'\d+[a-z]?', ref_clean)
            if not num_match:
                continue
            label = f"Figure {num_match.group(0)}"
            normalized = re.sub(r'[.\s]', '', label.lower())

            if normalized in found_labels:
                continue  # formal caption already exists

            # Extract the sentence containing this reference
            context_match = re.search(
                rf'([^.]*\b{re.escape(ref_clean)}\b[^.]*\.)',
                clean_text, re.IGNORECASE
            )
            if not context_match:
                continue
            context = context_match.group(1).strip()

            # Skip if this reads like a reference/citation, not a caption
            if _REFERENCE_PATTERNS.search(context):
                continue

            found_labels.add(normalized)
            page_captions.append({
                "label": label,
                "full_caption": context[:200],
                "normalized": normalized,
                "confidence": "low",
            })

        if page_captions:
            caption_map[page_num] = page_captions

    return caption_map


