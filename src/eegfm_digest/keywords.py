"""Keyword constants for arXiv retrieval."""

QUERY = (
    'all:(eeg OR electroencephalograph* OR brainwave*) AND '
    'all:("foundation model" OR pretrain OR pretrained OR '
    '"self-supervised" OR "self supervised" OR "representation learning" OR '
    'masked OR transfer OR generaliz*)'
)

# Display terms for the About/process page (must stay aligned with QUERY_A / QUERY_B).
EEG_KEYWORDS = [
    "eeg",
    "electroencephalograph*",
    "brainwave*",
]

FM_KEYWORDS_SET_A = [
    '"foundation model"',
    "pretrain",
    "pretrained",
    '"self-supervised"',
    '"self supervised"',
]

FM_KEYWORDS_SET_B = [
    '"representation learning"',
    "masked",
    "transfer",
    "generaliz*",
]
