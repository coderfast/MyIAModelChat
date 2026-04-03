import os
import warnings
import argparse
import logging
import importlib
import re
import psutil
import torch
from dialogmanager import DialogueManager
from simpletokenizer import SimpleTokenizer
from chatmodel import ChatModel
from transformers import pipeline

# Configuración principal
use_trusted = False  # Cambia a True solo si confías totalmente en el checkpoint
CKPT_PATH = 'chat_model.pth'
MAX_RAM_GB = None  # Ej: 4 para limitar a 4GB; None para no aplicar límite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

class MainChat:
    def __init__(self, args):
        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*", category=FutureWarning)

        self.chat = args.chat
        self.use_cpuonly = getattr(args, 'use_cpuonly', False)
        self.cuda_device = getattr(args, 'cuda_device', None)

        if self.cuda_device is not None and not self.use_cpuonly:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)
            logger.info("Using CUDA devices limited to: %s", self.cuda_device)

        device = self.get_device()

        # Optional memory limit
        if MAX_RAM_GB is not None:
            try:
                p = psutil.Process()
                mem_bytes = MAX_RAM_GB * 1024**3
                p.rlimit(psutil.RLIMIT_AS, (mem_bytes, mem_bytes))
                logger.info("Applied RAM limit: %s GB", MAX_RAM_GB)
            except Exception as e:
                logger.warning("Could not enforce RAM limit via psutil: %s", e)

        # Tokenizer and model
        self.tokenizer = None
        vocab_file = 'tokenizer_vocab.json'

        if os.path.exists(vocab_file):
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.load_vocabulary(vocab_file)
            logger.info(f"Loaded tokenizer vocabulary from {vocab_file}, vocab_size: {self.tokenizer.vocab_size}")

        try:
            ckpt = self.try_load_checkpoint(CKPT_PATH, device, use_trusted)
            if isinstance(ckpt, dict):
                # Load tokenizer from checkpoint if included
                if 'tokenizer' in ckpt and ckpt['tokenizer'] is not None:
                    self.tokenizer = ckpt['tokenizer']
                    logger.info(f"Loaded tokenizer from checkpoint, vocab_size: {self.tokenizer.vocab_size}")

                # Extract state dict for the model
                state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', None))
            else:
                state_dict = ckpt

            if self.tokenizer is None:
                self.tokenizer = SimpleTokenizer()
                logger.warning("No tokenizer found in checkpoint or vocab file. Using fresh SimpleTokenizer.")

            # Instantiate model with tokenizer-derived vocab
            self.model = ChatModel(self.tokenizer, embed_size=128, hidden_size=256)

            if state_dict is None:
                raise ValueError('Checkpoint does not contain model state dict')

            new_state = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                new_state[new_key] = v

            self.model.load_state_dict(new_state)
            self.model.to(device)
            self.model.eval()
            logger.info("Pre-trained model loaded successfully.")

        except Exception as e:
            logger.exception("Error loading pre-trained model: %s", e)
            if self.tokenizer is None:
                self.tokenizer = SimpleTokenizer()
            self.model = ChatModel(self.tokenizer, embed_size=128, hidden_size=256)
            logger.info("Initializing model with random weights...")
            init_fn = getattr(self.tokenizer, "init_weights", None)
            if callable(init_fn):
                self.model.apply(init_fn)
            else:
                def default_init(m):
                    if hasattr(m, "weight") and m.weight is not None:
                        try:
                            torch.nn.init.xavier_uniform_(m.weight)
                        except Exception:
                            pass
                    if hasattr(m, "bias") and m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                self.model.apply(default_init)
            self.model.to(device)
            self.model.eval()

        logger.info("Using device: %s", device)

        # Pipelines auxiliares
        intent_classifier = pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
        sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

        # DialogueManager configurable
        self.dialogue_manager = DialogueManager(
            model=self.model,
            device=device,
            tokenizer=self.tokenizer,
            intent_classifier=intent_classifier,
            sentiment_analyzer=sentiment_analyzer,
            persona={
                "name": "Eduardo Piñera Aznárez",
                "age": 51,
                "occupation": "AI assistant",
                "interests": ["IT technology", "MS Office", "Libre Office", "Games", "Humanity simulation"]
            },
            top_k=20,
            top_p=0.85,
            temperature=0.7,
            max_len=128,
            min_length=5,
            no_repeat_ngram_size=3,
            default_response="Lo siento, no puedo responder ahora."
        )

    def get_device(self):
        if self.use_cpuonly:
            logger.info("CPU-only mode enabled")
            return torch.device('cpu')
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                logger.info("Using CUDA: %s", name)
            except Exception:
                logger.info("Using CUDA")
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            logger.info("Using MPS")
            return torch.device('mps')
        logger.info("Using CPU")
        return torch.device('cpu')

    def try_load_checkpoint(self, path, device, trusted):
        try:
            if trusted:
                # Load complete checkpoint (includes tokenizer object)
                return torch.load(path, map_location=device, weights_only=False)
            else:
                # Prefer weights-only for safety, fallback if it fails for local trusted checkpoint
                try:
                    return torch.load(path, map_location=device, weights_only=True)
                except Exception:
                    return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            # Attempt to parse unsupported globals and allowlist them
            msg = str(e)
            m = re.search(r"Unsupported global: GLOBAL\s+([\w\.]+)\s+was", msg)
            if m:
                full_name = m.group(1)
                try:
                    module_name, class_name = full_name.rsplit('.', 1)
                    mod = importlib.import_module(module_name)
                    cls = getattr(mod, class_name)
                    with torch.serialization.safe_globals([cls]):
                        return torch.load(path, map_location=device, weights_only=False)
                except Exception:
                    logger.exception("Allowlisting failed for %s", full_name)
                    raise

            # Allow key My tokenizer types in case tokenizer object is stored
            try:
                import simpletokenizer
                allowed = [simpletokenizer.Trie, simpletokenizer.BilingualTokenizer, simpletokenizer.SimpleTokenizer]
                with torch.serialization.safe_globals(allowed):
                    return torch.load(path, map_location=device, weights_only=False)
            except Exception:
                pass

            raise

    def performMainChat(self):
        logger.info("Starting chat interface. Press ESC to exit.")
        try:
            while True:
                try:
                    user_input = input("You: ")
                except (EOFError, KeyboardInterrupt):
                    logger.info("Exiting chat loop.")
                    break

                if user_input.lower() in ('quit', 'exit'):
                    break

                response = self.dialogue_manager.generate_response(user_input)
                print("Bot:", response)
        except KeyboardInterrupt:
            logger.info("Chat terminated by user.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--use-cpuonly', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main = MainChat(args)
    if args.chat:
        main.performMainChat()
