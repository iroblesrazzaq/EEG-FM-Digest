"""Keyword and category constants for arXiv retrieval."""

ARXIV_CATEGORIES = {
    "q-bio.NC",
    "cs.LG",
    "stat.ML",
    "eess.SP",
    "cs.AI",
    "cs.NE",
}

QUERY_A = (
    'all:(eeg OR electroencephalograph* OR brainwave*) AND '
    'all:("foundation model" OR pretrain OR pretrained OR "self-supervised" OR "self supervised")'
)

QUERY_B = (
    'all:(eeg OR electroencephalograph* OR brainwave*) AND '
    'all:("representation learning" OR masked OR transfer OR generaliz*)'
)
