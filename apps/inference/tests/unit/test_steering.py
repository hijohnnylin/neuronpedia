from neuronpedia_inference.inference_utils.steering import apply_generic_chat_template


def test_single_message_with_generation_prompt():
    """Test basic functionality with generation prompt."""
    messages = [{"role": "user", "content": "Hello"}]
    result = apply_generic_chat_template(messages, add_generation_prompt=True)
    expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    assert result == expected


def test_single_message_without_generation_prompt():
    """Test without generation prompt."""
    messages = [{"role": "user", "content": "Hello"}]
    result = apply_generic_chat_template(messages, add_generation_prompt=False)
    expected = "<|im_start|>user\nHello<|im_end|>\n"
    assert result == expected


def test_multiple_messages():
    """Test multiple message conversation."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = apply_generic_chat_template(messages)
    expected = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n<|im_start|>assistant\n"
    assert result == expected


def test_empty_messages():
    """Test empty message list."""
    result = apply_generic_chat_template([])
    expected = "<|im_start|>assistant\n"
    assert result == expected
