import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_train
from simpletokenizer import SimpleTokenizer


def test_load_vocabulary_ignores_sentencepiece_metadata(tmp_path):
    tokenizer = SimpleTokenizer()
    metadata_path = tmp_path / "tokenizer_vocab.json"
    metadata_path.write_text(
        json.dumps({"sentencepiece_model": "dataset_cache/sentencepiece.model", "vocab_size": 2000}),
        encoding="utf-8",
    )

    tokenizer.load_vocabulary(str(metadata_path))

    assert tokenizer.get_index("<PAD>") == 0
    assert tokenizer.get_index("<UNK>") == 1
    assert tokenizer.get_index("<START>") == 2
    assert tokenizer.get_index("<END>") == 3
    assert tokenizer.vocab_size == 4


def test_sentencepiece_wrapper_decodes_ids_to_text():
    class FakeSentencePieceProcessor:
        def load(self, path):
            self.path = path

        def piece_to_id(self, piece):
            mapping = {"<pad>": 0, "<PAD>": 0, "<unk>": 1, "<UNK>": 1, "hello": 2, "world": 3}
            return mapping.get(piece, -1)

        def encode(self, text, out_type=int):
            return [2, 3]

        def get_piece_size(self):
            return 4

        def decode_ids(self, ids):
            return "hello world"

        def id_to_piece(self, token_id):
            return {0: "<pad>", 1: "<unk>", 2: "hello", 3: "world"}.get(token_id, str(token_id))

    main_train.spm = types.SimpleNamespace(SentencePieceProcessor=FakeSentencePieceProcessor)
    tokenizer = main_train.SentencePieceTokenizerWrapper("fake.model")

    assert tokenizer.decode([2, 3]) == "hello world"
    assert tokenizer.convert_ids_to_tokens([2, 3]) == ["hello", "world"]
