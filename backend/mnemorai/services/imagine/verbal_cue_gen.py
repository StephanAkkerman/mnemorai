import asyncio

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from mnemorai.constants.config import config
from mnemorai.logger import logger
from mnemorai.services.pre.grapheme2phoneme import Grapheme2Phoneme
from mnemorai.services.pre.translator import translate_word
from mnemorai.utils.lang_codes import map_language_code
from mnemorai.utils.load_models import select_model
from mnemorai.utils.model_mem import manage_memory
from mnemorai.utils.ngram_probs import ngrams_to_df
from mnemorai.utils.syllabifier import syllables


def _mnemonic_prompt_template(
    word: str, n: int = config.get("LLM").get("MNEMONIC_CANDIDATES")
):
    return f"""
Word: {word}
Task: Brainstorm {n} ENGLISH words that sounds similar to "{word}"
Guidelines:  
- Loose phonetic similarity is welcome.  
- You may swap or drop consonants/vowels (e.g., 'd'→'th', 'g'→'k', etc.).  
- Voicing changes and soft consonant substitutions (e.g., 'd' ↔ 'th', 'b' ↔ 'v') are encouraged.  
- Function words, articles, and short high-frequency words are allowed.  
- Only include **standard, commonly used English words** (avoid slang, interjections, contractions, or non-words).  
- Try not to use many extra letters than the given word.  
- No translations, no definitions—just comma-separated list, lower-case.  
- Start with the most suitable candidates first.  
- Only include words with similar sounds, not synonyms or related concepts.  

Return: one line of comma-separated candidates, no explanation.
"""


def _mnemonic_selector_template(
    word: str, meaning: str, candidates: list[str], n: int = 25
):
    return f"""
What is a fitting mnemonic for the word "{word}" (meaning {meaning}). 
Choose from the following list and only respond with the result.

The candidates are: {', '.join(candidates[:n])}.
"""


def _verbal_cue_prompt_template(word1: str, word2: str):
    return f"""
I want a mnemonic to combine "{word1}" and "{word2}" think of something that I can visualize. 
The output will be used as a prompt for image generation, so it should be a vivid and memorable sentence.
"""


class VerbalCue:
    def __init__(self, model_name: str = None):
        self.config = config.get("LLM")
        self.offload = self.config.get("OFFLOAD")
        self.model_name = model_name if model_name else select_model(self.config)
        logger.debug(f"Selected LLM model: {self.model_name}")

        self.g2p_model = Grapheme2Phoneme()
        self.generation_args = self.config.get("PARAMS")

        logger.debug(f"LLM generation args: {self.generation_args}")

        self.mnemonic_messages = [
            {
                "role": "system",
                "content": """You are a phonetic pun generator.""",
            },
            {
                "role": "user",
                "content": _mnemonic_prompt_template("flasche"),
            },
            {
                "role": "assistant",
                "content": """flashy, flash, flask, flasher, fleshy, flusher, flush, flash he, flesh he, fleshy""",
            },
        ]

        self.verbal_cue_messages = [
            {
                "role": "system",
                "content": """
You are an imaginative and detail-oriented visual scene generator. Your job is to take two given concepts and craft a vivid, coherent prompt suitable for AI image generation.
For each prompt:
    - Blend both concepts into a single, creative scene
    - Use rich, visual language that evokes strong mental imagery
    - Describe characters, setting, lighting, mood, and key elements
    - Focus on being clear, cinematic, and image-friendly — no backstory, no abstract explanations
    - If appropriate, inject a sense of whimsy, surrealism, or narrative tension
""",
            },
            {
                "role": "user",
                "content": _verbal_cue_prompt_template("bottle", "flashy"),
            },
            {
                "role": "assistant",
                "content": "A dazzling glass bottle standing on a velvet pedestal in a dark room, illuminated by intense, colorful spotlights. The bottle is encrusted with glittering jewels, glowing neon patterns, and emits radiant sparkles as if it's alive. Around it, confetti floats in the air, and golden beams shoot upward like it's at the center of a high-end fashion show or magic performance. The setting is glamorous and surreal, with reflections and lens flares adding to the flashy spectacle.",
            },
        ]
        # This will be initialized later
        self.pipe = None
        self.tokenizer = None  # Initialize tokenizer attribute
        self.model = None  # Initialize model attribute

    def _initialize_pipe(self):
        """Initialize the pipeline."""
        logger.debug(f"Initializing pipeline for LLM with model: {self.model_name}")

        bnb_config = None
        q = config.get("LLM", {}).get("QUANTIZATION")
        if q == "4bit":
            logger.debug("Using 4-bit quantization for LLM")
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        elif q == "8bit":
            logger.debug("Using 8-bit quantization for LLM")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # build up a dict of kwargs
        kwargs = {
            "device_map": "cuda" if self.offload else "auto",
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "cache_dir": "models",
        }
        # only include quantization_config if we actually have one
        if bnb_config is not None:
            kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        # Check if LoRA should be enabled
        if config.get("LLM").get("USE_LORA"):
            lora = config.get("LLM").get("LORA")
            logger.debug(f"Loading LoRA ({lora}) for LLM")
            # Load the model with LoRA
            self.model = PeftModel.from_pretrained(self.model, lora)
            self.model = self.model.merge_and_unload()

        # Ensure the model is in evaluation mode (disables dropout, etc.)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="models",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def get_candidates(self, word: str):
        final_message = {
            "role": "user",
            "content": _mnemonic_prompt_template(word=word),
        }

        logger.debug(f"Generating mnemonics for word: {word}")
        # TODO: apply chat template
        output = self.pipe(
            self.mnemonic_messages + [final_message], **self.generation_args
        )
        response = output[0]["generated_text"]
        logger.debug(f"Generated mnemonics: {response[-1]['content']}")

        # parse the string into Python objects and find best match
        candidates = response[-1]["content"].strip().split(",")

        return candidates

    def get_best_candidate(self, word: str, language_code: str):
        # Split word into syllables
        word_syllables = syllables(word, language_code)
        logger.debug(f"Syllables for '{word}': {word_syllables}")

        # Also do this for the whole word
        full_word_candidates = self.get_candidates(word)

        syllable_candidates = []

        if word_syllables is not None:
            for syllable in word_syllables:
                syllable_candidates.append(self.get_candidates(syllable))

            # Get the n-gram probabilities for the candidates
            probs = ngrams_to_df([full_word_candidates], syllable_candidates)

        else:
            probs = ngrams_to_df([full_word_candidates])

        logger.debug(f"Probabilities DataFrame:\n{probs}")

        # Get the candidates
        candidates = probs["ngram"].tolist()

        # Ask the LLM for the best candidate

    @manage_memory(
        targets=["model"], delete_attrs=["model", "pipe", "tokenizer"], move_kwargs={}
    )
    async def generate_mnemonic(
        self,
        word: str,
        language_code: str,
        keyword: str = None,
        key_sentence: str = None,
    ) -> tuple:
        """
        Generate a mnemonic for the input word using the phonetic representation.

        Parameters
        ----------
        word : str
            Foreign word to generate mnemonic for.
        language_code : str
            Language code of the input word.
        keyword : str, optional
            User-provided keyword to use in the mnemonic.
        key_sentence : str, optional
            User-provided key sentence to use as the mnemonic.

        Returns
        -------
        tuple
            A tuple containing the top matches, translated word, transliterated word, IPA, and verbal cue.
        """
        # Convert the input word to IPA representation
        ipa = self.g2p_model.word2ipa(word=word, language_code=language_code)

        # Get the 2-letter language code
        language_code = map_language_code(language_code)
        logger.debug(f"Mapped language code: {language_code}")

        # Translate and transliterate the word
        translated_word, transliterated_word = await translate_word(word, language_code)

        # If a keyword or key sentence is provided do not generate mnemonics
        if not (keyword or key_sentence):
            best = self.get_best_candidate(word, language_code)

        if key_sentence:
            return best, translated_word, transliterated_word, ipa, key_sentence

        # Generate the verbal cue
        verbal_cue = self.generate_cue(
            word1=translated_word,
            word2=keyword if keyword else best,
        )

        return best, translated_word, transliterated_word, ipa, verbal_cue

    def generate_cue(self, word1: str, word2: str) -> str:
        """
        Generate a verbal cue that connects two words.

        Parameters
        ----------
        word1 : str
            The first word.
        word2 : str
            The second word.

        Returns
        -------
        str
            The generated verbal cue.
        """
        final_message = {
            "role": "user",
            "content": f"Generate a mnemonic sentence for the given input. Start the sentence with 'imagine' and keep it simple.\nInput: English Word: {word1} | Mnemonic Word: {word2}",
        }
        # For some reason using tokenizer.apply_chat_template() here causes weird output
        output = self.pipe(
            self.verbal_cue_messages + [final_message], **self.generation_args
        )
        response = output[0]["generated_text"]
        verbal_cue = response[-1]["content"]
        logger.debug(f"Generated verbal cue: {verbal_cue}")

        return verbal_cue


if __name__ == "__main__":
    vc = VerbalCue()
    print(asyncio.run(vc.generate_mnemonic(word="daging", language_code="ind")))
    # print(vc.generate_cue("bottle", "flashy"))
