from typing import Literal

import epitran
import pycountry
import pyphen
from lingpy.sequence.sound_classes import ipa2tokens, syllabify

from mnemorai.constants.languages import EPITRAN_LANGCODES
from mnemorai.logger import logger

# ---------- tiny registry ----------
_PYPHEN_LANGS = {k.split("_")[0]: k for k in pyphen.LANGUAGES}  # "id" → "id_ID"
_EPI_CACHE: dict[str, epitran.Epitran] = {}  # hold compiled epitran objects


def iso1_to_epi_tag(
    iso_code: str,
    *,
    default_script: Literal[
        "Latn",
        "Cyrl",
        "Arab",
        "Deva",
        "Ethi",
        "Hrgn",
        "Ktkn",
        "Hang",
        "Mlym",
        "Sinh",
        "Thai",
        "Guru",
        "Hebr",
    ] = "Latn",
) -> str:
    """
    Convert a 2-letter ISO 639-1 code (or ISO 639-1 code with script/region suffix) to the “EPI tag” format (e.g. "eng-Latn", "fra-Latn", "spa-Latn", etc.).

    Parameters
    ----------
    iso_code : str
        A 2-letter ISO 639-1 language code, optionally followed by a script or region subtag
        (e.g. "sr" or "sr-latn", "zh" or "zh-tw").
    default_script : Literal["Latn", "Cyrl", "Arab", "Deva", "Ethi", "Hrgn", "Ktkn", "Hang", "Mlym", "Sinh", "Thai", "Guru", "Hebr"], optional
        The default script to append if no override is found. Defaults to "Latn".

    Returns
    -------
    str
        The corresponding EPI tag (ISO 639-3 code + “-” + script), e.g. "eng-Latn", "deu-Latn", "nld-Latn".

    Raises
    ------
    LookupError
        If the given ISO 639-1 code cannot be mapped to an ISO 639-3 code via `pycountry`.
    """
    key = iso_code.strip().lower()

    # 2) Otherwise, try to split out any region/script subtag (e.g. "sr-Latn" or "en-US")
    base, _ = key.split("-")

    # 3) Lookup ISO3 via pycountry
    try:
        language = pycountry.languages.get(alpha_2=base)
        if not language or not hasattr(language, "alpha_3"):
            raise LookupError(f"Cannot find ISO3 for '{base}'")
        iso3 = language.alpha_3
    except Exception as e:
        raise LookupError(f"Error mapping '{iso_code}' to ISO3: {e}")

    # 4) Construct the tag using default_script (e.g. "eng-Latn")
    return f"{iso3}-{default_script}"


def _map_to_epitran_lang(lang: str) -> str:
    tag = lang.replace("_", "-")
    if len(tag.split("-")[0]) == 3:
        return tag  # assume “fra-Latn” or “ind-Latn”
    iso1 = tag.split("-")[0].lower()
    epitag = iso1_to_epi_tag(iso1)
    if epitag is None:
        raise ValueError(
            f"No Epitran mapping for “{lang}”; please supply “ind-Latn”, etc."
        )
    return epitag


def _align_ipa_to_orth(
    ipa_syllables: list[str], word: str, epi: epitran.Epitran
) -> list[str]:
    """
    Given.

      • ipa_syllables = ["sə","ma","ŋat"]
      • word = "semangat"
      • epi = Epitran("ind-Latn")
    Try to find, for each ipa_chunk, the shortest substring of word[idx:] whose epi.transliterate(...) == ipa_chunk.
    If none is found, fall back to returning ipa_chunk itself.
    """
    orth_sylls: list[str] = []
    idx = 0
    n = len(word)

    import unicodedata

    def _strip_tones(s: str) -> str:
        # remove diacritics (tone marks) from pinyin before Epitran
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    for ipa_chunk in ipa_syllables:
        matched = False
        # Try all possible substrings starting at idx, increasing end.
        for end in range(idx + 1, n + 1):
            candidate = word[idx:end]
            # strip tone marks before asking Epitran
            base = _strip_tones(candidate)
            translit = epi.transliterate(base)
            print(translit, ipa_chunk)
            # accept either a prefix‐match or a suffix‐match
            if translit.startswith(ipa_chunk) or ipa_chunk.endswith(translit):
                orth_sylls.append(candidate)
                idx = end
                matched = True
                break
        if not matched:
            # no perfect match → just return the IPA chunk itself
            orth_sylls.append(ipa_chunk)

    return orth_sylls


def _epitran_split(word: str, lang: str) -> list[str] | None:
    """
    Fallback syllabifier via Epitran → IPA → LingPy → brute-force align back to orthography.

    1) Map “id”→“ind-Latn”, etc.; load epi = Epitran(epi_tag).
    2) ipa = epi.transliterate(word)
    3) tokens = ipa2tokens(ipa, merge_vowels=True, merge_diphthongs=True)
    4) raw = syllabify(tokens)  → maybe nested lists or flat+'+'
    5) turn raw into list of IPA strings, e.g. ["sə","ma","ŋat"]
    6) call _align_ipa_to_orth(ipa_syllables, word, epi) to get ["se","ma","ngat"]
    """
    # 1) figure out the correct tag for Epitran (“ind-Latn” instead of “id”)
    try:
        epi_tag = _map_to_epitran_lang(lang)
    except ValueError:
        logger.warning(
            f"Unsupported language code '{lang}' for Epitran syllabification."
        )
        return

    # Check if this tag is supported by Epitran
    if epi_tag not in EPITRAN_LANGCODES:
        logger.warning(f"Epitran does not support the language code '{epi_tag}'. ")
        return

    if not epi_tag.endswith("-Latn"):
        logger.warning(
            f"Mnemorai syllabification is only supported for languages with a Latin script. "
            f"'{epi_tag}' does not end with '-Latn'."
        )
        return

    # 2) load or retrieve from cache
    try:
        epi = _EPI_CACHE.setdefault(epi_tag, epitran.Epitran(epi_tag))
    except OSError as exc:
        logger.warning(f"Failed to load Epitran for '{epi_tag}': {exc}. ")

    # 3) transliterate to IPA (e.g. "səmaŋat")
    ipa = epi.transliterate(word)

    # 4) tokenize IPA into segments
    tokens = ipa2tokens(ipa, merge_vowels=True, merge_diphthongs=True)
    #    ↳ e.g. ['s','ə','m','a','ŋ','a','t']

    # 5) syllabify(tokens) → either nested or flat+‘+’
    try:
        raw = syllabify(tokens)
    except Exception as exc:
        logger.warning(f"Failed to syllabify '{word}' with Epitran: {exc}. ")

    # 5a) normalize to list-of-lists
    if raw and isinstance(raw[0], list):
        syl_token_lists: list[list[str]] = raw  # type: ignore
    else:
        flat = raw  # type: ignore
        syl_token_lists = []
        current: list[str] = []
        for seg in flat:
            if seg == "+":
                if current:
                    syl_token_lists.append(current)
                current = []
            else:
                current.append(seg)
        if current:
            syl_token_lists.append(current)

    # 6) join each sub-list into a pure IPA string
    ipa_syllables = ["".join(s) for s in syl_token_lists]  # e.g. ["sə","ma","ŋat"]

    # 7) align each IPA chunk back to a substring of “word”
    return _align_ipa_to_orth(ipa_syllables, word, epi)


def _pyphen_split(word: str, lang: str) -> list[str] | None:
    """Try Pyphen hyphenation; return None if language unsupported."""
    code = _PYPHEN_LANGS.get(lang.split("-")[0])
    if not code:
        logger.debug(
            f"Pyphen does not support the language code '{lang}'. "
            "Falling back to Epitran syllabification."
        )
        return None
    return pyphen.Pyphen(lang=code).inserted(word).split("-")


def syllables(word: str, lang: str) -> list[str]:
    """
    Return a list of syllables for *word* written in *lang* (ISO-639 code).

    Parameters
    ----------
    word : str
        The token to split.
    lang : str
        ISO-639-1 or BCP-47 code understood by Pyphen and/or Epitran
        (e.g. 'id', 'es', 'fr', 'zh-Hans').

    Examples
    --------
    >>> syllables("daging", "id")
    ['da', 'ging']
    >>> syllables("computer", "en")
    ['com', 'pu', 'ter']
    """
    # Get the best result
    return pick_best(_pyphen_split(word, lang), _epitran_split(word, lang))


def pick_best(sol1: list[str], sol2: list[str]) -> list[str]:
    """
    Return whichever solution has fewer syllable-chunks.

    If they're equal length, return sol1 by default.
    """
    if sol1 is None:
        return sol2
    if sol2 is None:
        return sol1

    if sol1 is None and sol2 is None:
        print("Warning: both syllable solutions are None; returning empty list.")
        return []

    return sol1 if len(sol1) <= len(sol2) else sol2


if __name__ == "__main__":
    print(syllables("semangat", "id"))  # → ['se', 'ma', 'ngat]
    print(syllables("bonjour", "fr"))  # → ['bon', 'jour']
    print(syllables("fiesta", "es"))  # → ['fies', 'ta']
    print(syllables("schönheit", "de"))  # → ['schön', 'heit']
    print(syllables("hondsdolheid", "nl"))  # → ['honds', 'dol', 'heid']
