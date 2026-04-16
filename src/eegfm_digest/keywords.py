"""Keyword and category constants for arXiv retrieval."""

ARXIV_CATEGORY_LIST = (
    "q-bio.NC",
    "cs.LG",
    "stat.ML",
    "eess.SP",
    "cs.AI",
    "cs.NE",
)
ARXIV_CATEGORIES = set(ARXIV_CATEGORY_LIST)

KEYWORD_ANCHOR_TERMS = (
    "eeg",
    "electroencephalograph*",
    "brainwave*",
)

KEYWORD_QUERY_A_TERMS = (
    '"foundation model"',
    "pretrain",
    "pretrained",
    '"self-supervised"',
    '"self supervised"',
)

KEYWORD_QUERY_B_TERMS = (
    '"representation learning"',
    "masked",
    "transfer",
    "generaliz*",
)


def _build_query(anchor_terms: tuple[str, ...], target_terms: tuple[str, ...]) -> str:
    return f"all:({' OR '.join(anchor_terms)}) AND all:({' OR '.join(target_terms)})"


QUERY_A = _build_query(KEYWORD_ANCHOR_TERMS, KEYWORD_QUERY_A_TERMS)
QUERY_B = _build_query(KEYWORD_ANCHOR_TERMS, KEYWORD_QUERY_B_TERMS)
