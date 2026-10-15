"""Character set of the swissdocai word recognizer.

docTR's pretrained recognizers use a French vocabulary: no ä, ö, ß, no
typographic apostrophe, guillemets or dashes — they cannot read most German
text. The Swiss vocabulary adds German, Italian and Romansh letters and the
punctuation of Swiss official and business documents.
"""

_FRENCH = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"
)
_ADDITIONS = (
    "äöÄÖßẞ"  # German
    "ìíòóúÌÍÒÓÚáÁ"  # Italian, Romansh
    "œŒæÆñÑ"
    "’‘“”„«»‹›"  # typographic quotes
    "–—…"  # dashes, ellipsis
    "§±×÷µ²³¹·•½¼¾"
)

SWISS_VOCAB = "".join(dict.fromkeys(_FRENCH + _ADDITIONS))
