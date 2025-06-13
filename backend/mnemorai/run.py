import asyncio

import torch

from mnemorai.logger import logger
from mnemorai.services.imagine.image_gen import ImageGen
from mnemorai.services.imagine.verbal_cue_gen import VerbalCue
from mnemorai.services.tts.tts import TTS


class MnemonicPipeline:
    def __init__(self):
        # Check if cuda is available
        logger.info(f"cuda available: {torch.cuda.is_available()}")
        logger.info(f"cuda device count: {torch.cuda.device_count()}")

    async def generate_mnemonic_img(
        self,
        word: str,
        lang_code: str,
        llm_model: str = None,
        image_model: str = None,
        keyword: str = None,
        key_sentence: str = None,
    ) -> tuple:
        """
        Generate an image for a given word using the mnemonic pipeline.

        Parameters
        ----------
        word : str
            The word to generate an image for in the language of lang_code.
        lang_code : str
            The language code for the word.
        llm_model : str, optional
            The name of the LLM model to use for verbal cue generation.
        image_model : str, optional
            The name of the image model to use for image generation.

        Returns
        -------
        str
            The path to the generated image.
        str
            The verbal cue for the image.
        str
            The translated word.
        str
            The path to the generated audio file.
        str
            The IPA spelling of the best match.
        """
        # Use the provided llm_model if available, otherwise default to the one in config
        if llm_model:
            vc = VerbalCue(model_name=llm_model)
        else:
            vc = VerbalCue()

        # Generate a verbal cue
        logger.debug(f"Generating verbal cue for '{word}'...")
        _, translated_word, _, ipa, key_sentence = await vc.generate_mnemonic(
            word=word,
            language_code=lang_code,
            keyword=keyword,
            key_sentence=key_sentence,
        )

        # Use the provided image_model if available, otherwise default to the one in config
        if image_model:
            img_gen = ImageGen(model=image_model)
        else:
            img_gen = ImageGen()

        # Generate the image
        image_path = img_gen.generate_img(
            prompt=key_sentence, word1=word, word2=keyword
        )

        # Generate TTS
        tts_model = TTS()
        tts_path = tts_model.tts(word, lang=lang_code)

        return image_path, key_sentence, translated_word, tts_path, ipa


if __name__ == "__main__":
    pipeline = MnemonicPipeline()
    print(asyncio.run(pipeline.generate_mnemonic_img("tikus", "ind")))
