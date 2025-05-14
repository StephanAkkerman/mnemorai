import ast
import asyncio
import re

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from mnemorai.constants.config import config
from mnemorai.constants.languages import G2P_LANGUAGES
from mnemorai.logger import logger
from mnemorai.services.pre.grapheme2phoneme import Grapheme2Phoneme
from mnemorai.services.pre.translator import translate_word
from mnemorai.utils.load_models import select_model
from mnemorai.utils.model_mem import manage_memory


def _parse_mnemonic_list(model_output: str):
    # 1. grab the first [...] block
    m = re.search(r"\[.*\]", model_output, re.DOTALL)
    if not m:
        raise ValueError("No list literal found in response")
    list_str = m.group(0)

    # 2. safely evaluate it into Python objects
    options = ast.literal_eval(list_str)

    # 3. pick the mnemonic with the highest score
    best = max(options, key=lambda opt: opt["score"])
    return best["mnemonic"]


class VerbalCue:
    def __init__(self, model_name: str = None):
        self.config = config.get("LLM")
        self.offload = self.config.get("OFFLOAD")
        self.model_name = model_name if model_name else select_model(self.config)
        logger.debug(f"Selected LLM model: {self.model_name}")
        self.g2p_model = Grapheme2Phoneme()

        # Use 512 tokens as default for generation
        self.generation_args = {
            "max_new_tokens": 512,
        }
        # Add the config parameters
        config_params = self.config.get("PARAMS")
        if config_params:
            self.generation_args.update(config_params)

        logger.debug(f"LLM generation args: {self.generation_args}")

        self.mnemonic_messages = [
            {
                "role": "system",
                "content": """
You are a creative mnemonic-generation assistant. For any foreign word input, produce exactly 10 English mnemonic words or short phrases that:

1. Are phonetically similar to the original word
2. Look orthographically similar
3. Are semantically related or evoke a relevant image
4. Are highly imageable and easy to imagine

Score each mnemonic from 0.00 to 1.00 using weights: phonetic_similarity=0.4, orthographic_similarity=0.3, semantic_relatedness=0.2, imageability=0.1. Sort the list in descending order by score.

Output a JSON list of objects (use double quotes) with keys "mnemonic" and "score", exactly like:
[
  {"mnemonic": "...", "score": 0.95},
  ...
]

Do not translate the word or include extra commentary; focus solely on generating memorable, natural-sounding mnemonics.
                """,
            },
            {
                "role": "user",
                "content": """Generate mnemonics to remember the German word 'flasche'.""",
            },
            {
                "role": "assistant",
                "content": """
                [
                    {'mnemonic': 'flashy', 'score': 0.91},
                    {'mnemonic': 'flash', 'score': 0.89},
                    {'mnemonic': 'flask', 'score': 0.87},
                    {'mnemonic': 'flasher', 'score': 0.85},
                    {'mnemonic': 'fleshy', 'score': 0.82},
                    {'mnemonic': 'flusher', 'score': 0.78},
                    {'mnemonic': 'flush', 'score': 0.76},
                    {'mnemonic': 'flash he', 'score': 0.65},
                    {'mnemonic': 'flesh he', 'score': 0.60},
                    {'mnemonic': 'flossier', 'score': 0.55},
                ]
                """,
            },
        ]

        self.verbal_cue_messages = [
            {
                "role": "system",
                "content": "You are a creative mnemonic-cue assistant. When given a target word and its chosen mnemonic, generate a short, vivid, and memorable sentence that starts with 'Imagine', uses both words, and creates a clear mental image to link them. Keep it simple and catchy.",
            },
            {
                "role": "user",
                "content": "Generate a mnemonic cue for the input.\nInput: English Word: bottle | Mnemonic Word: flashy",
            },
            {
                "role": "assistant",
                "content": "Imagine a flashy bottle sparkling under neon lights, impossible to ignore.",
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

        # Translate and transliterate the word
        translated_word, transliterated_word = await translate_word(word, language_code)

        # If a keyword or key sentence is provided do not generate mnemonics
        if not (keyword or key_sentence):
            language = G2P_LANGUAGES.get(language_code, "Unknown Language")
            final_message = {
                "role": "user",
                "content": f"""Generate mnemonics to remember the {language} word '{word}'.""",
            }

            output = self.pipe(
                self.mnemonic_messages + [final_message], **self.generation_args
            )
            response = output[0]["generated_text"]
            logger.debug(f"Generated mnemonics: {response[-1]['content']}")

            # parse the string into Python objects and find best match
            best = _parse_mnemonic_list(response[-1]["content"])

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
