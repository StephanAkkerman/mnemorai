import asyncio
import unicodedata

import pandas as pd
from googletrans import Translator
from httpx import Timeout

from mnemorai.logger import logger
from mnemorai.utils.lang_codes import map_language_code

shared_translator = Translator(timeout=Timeout(None), list_operation_max_concurrency=50)


def is_latin_script(word: str) -> bool:
    """
    Checks if the given word is written in the Latin script.

    Parameters
    ----------
    word : str
        The word to check.

    Returns
    -------
    bool
        True if the word is written in the Latin script, False otherwise.
    """
    for char in word:
        if "LATIN" not in unicodedata.name(char):
            return False
    return True


def remove_diacritics(word: str) -> str:
    """
    Removes diacritics from a word.

    Parameters
    ----------
    word : str
        The word to remove diacritics from.

    Returns
    -------
    str
        The word without diacritics.
    """
    # Normalize the word to decomposed form (NFD)
    decomposed = unicodedata.normalize("NFD", word)
    # Filter out diacritic characters (those in the 'Mn' category)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


async def get_transliteration(word: str, src: str) -> str:
    """
    Given a word, returns its transliteration in the same language.

    Parameters
    ----------
    word : str
        The word to transliterate.
    src : str
        The language code of the word.

    Returns
    -------
    str
        The transliterated word.
    """
    if is_latin_script(word):
        return word

    transliterated_word = await get_translation(
        word, src, src, return_pronunciation=True
    )

    # Lower case it
    transliterated_word = transliterated_word.lower()

    # Remove diacritics
    return remove_diacritics(transliterated_word)


async def get_translation(
    word: str, src: str, target: str, return_pronunciation: bool = False
) -> str:
    """
    Translates a word from the source language to the target language.

    Parameters
    ----------
    word : str
        The word in the source (src) language.
    src : str
        The language code of the source language.
    target : str
        The language code of the target language.

    Returns
    -------
    str
        The translated word.
    """
    logger.debug(f"Getting the translation for {word} from {src} to {target}...")
    try:
        async with Translator() as shared_translator:
            translation = await shared_translator.translate(word, src=src, dest=target)
            if return_pronunciation:
                return translation.pronunciation
            return translation.text
    except Exception as e:
        logger.error(f"Error translating {word} from {src} to {target}: {e}")
        # Show full traceback
        logger.debug(e, exc_info=True)

    # Detect the language of the word
    async with Translator() as shared_translator:
        detected_lang = await shared_translator.detect(word)
        logger.info(
            f"Could not comprehend original language code ({src}), detected {detected_lang.lang} with {detected_lang.confidence} confidence."
        )

        translation = await shared_translator.translate(
            word, src=detected_lang.lang, dest=target
        )
        if return_pronunciation:
            return translation.pronunciation
        return translation.text


async def translate_word(word, src_lang_code, target_lang_code: str = "en") -> tuple:
    """
    Translates a word from the source language to target languages.

    Args:
        word (str): The word to translate.
        src_lang_code (str): The source language code (ISO 639-1 or custom).
        target_lang_code (str): A target language codes (ISO 639-1 or custom).

    Returns
    -------
        dict: A dictionary with target language codes as keys and translated words as values.
    """
    logger.debug(f"Translating {word} from {src_lang_code} to {target_lang_code}...")

    if len(src_lang_code) > 2:
        src = map_language_code(src_lang_code)
        logger.debug(
            f"Mapped translation source language code: {src_lang_code} -> {src}"
        )
    else:
        src = src_lang_code

    # Map target language codes.
    if len(target_lang_code) > 2:
        target = map_language_code(target_lang_code)
        logger.debug(
            f"Mapped translation target language code: {target_lang_code} -> {target}"
        )
    else:
        target = target_lang_code

    return await get_translation(word, src, target), await get_transliteration(
        word, src
    )


async def translate_dataframe_column(
    df: pd.DataFrame,
    word_col: str,
    lang_code: str,
    translated_col: str = "translated_word",
    target_lang_code: str = "en",
) -> pd.DataFrame:
    """
    Asynchronously translates a column in the DataFrame to the target language.

    Args:
        df (pd.DataFrame): The DataFrame containing words and language codes.
        word_col (str): The name of the column containing words to translate.
        lang_code (str): The source language code.
        translated_col (str, optional): The name of the new column for translations. Defaults to 'translated_word'.
        target_lang_code (str, optional): The target language code. Defaults to 'en'.

    Returns
    -------
        pd.DataFrame: The DataFrame with an added translated column.
    """

    # Define a coroutine for translating a single word
    async def translate_row(word):
        return await translate_word(word, lang_code, target_lang_code)

    # Create a list of coroutines for all words to translate
    tasks = [translate_row(word) for word in df[word_col]]

    # Alternatively, without a progress bar:
    translations = await asyncio.gather(*tasks)

    # Assign the translations to the new column in the DataFrame
    df[translated_col] = translations

    return df


if __name__ == "__main__":
    print(asyncio.run(translate_word("kat", "dut")))
    print(asyncio.run(translate_word("amigo", "spa-latin")))
    print(asyncio.run(translate_word("猫", "zho-s")))

    # Test the DataFrame translation
    df = pd.DataFrame({"word": ["kat", "hond", "kip"]})
    print(asyncio.run(translate_dataframe_column(df, "word", "dut")))
