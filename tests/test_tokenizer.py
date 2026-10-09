from llm_from_scratch.tokenizer import (
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
