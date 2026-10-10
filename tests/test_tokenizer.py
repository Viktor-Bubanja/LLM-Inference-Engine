import tokenizers

from llm_from_scratch.tokenizer import (
    HFTokenizer,
    TiktokenTokenizer,
    get_tokenizer,
    text_to_token_ids,
    token_ids_to_text,
)


def test_text_round_trips_through_a_batch_of_one():
    tokenizer = get_tokenizer()
    text = "Every effort moves you"

    token_ids = text_to_token_ids(text, tokenizer)

    assert token_ids.shape == (1, len(tokenizer.encode(text)))
    assert token_ids_to_text(token_ids, tokenizer) == text


def test_tiktoken_tokenizer_exposes_gpt2_special_ids():
    tokenizer = TiktokenTokenizer()
    text = "Every effort moves you"

    assert tokenizer.eos_id == 50256
    assert tokenizer.vocab_size == 50257
    assert tokenizer.decode(tokenizer.encode(text)) == text


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
