from llm_from_scratch.chat import stream_text


def test_streamed_text_never_splits_a_character(byte_chat_format):
    tokenizer = byte_chat_format.tokenizer
    text = "Grüße aus Perú 🦙!"

    pieces = list(stream_text(tokenizer, tokenizer.encode(text)))

    assert "".join(pieces) == text
    assert not any("\ufffd" in piece for piece in pieces)
