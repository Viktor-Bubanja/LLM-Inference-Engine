import tokenizers

from llm_from_scratch.tokenizer import HFTokenizer


def test_hf_tokenizer_round_trips_without_adding_special_tokens(tmp_path):
    vocab = {"[UNK]": 0, "hello": 1, "alpaca": 2, "<eos>": 3}
    hf_tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]")
    )
    hf_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    hf_tokenizer.save(str(tmp_path / "tokenizer.json"))

    tokenizer = HFTokenizer(tmp_path / "tokenizer.json", eos_id=3)

    assert tokenizer.encode("hello alpaca") == [1, 2]
    assert tokenizer.decode([1, 2]) == "hello alpaca"
    assert tokenizer.eos_id == 3
    assert tokenizer.vocab_size == 4
