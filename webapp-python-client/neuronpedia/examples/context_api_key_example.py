"""
Example demonstrating the new context-based API key management using contextvars.
This provides a much cleaner API compared to passing api_key to every constructor.
"""

import neuronpedia


def example_context_manager():
    """Using context manager (recommended for temporary scope)"""
    print("=== Context Manager Example ===")

    with neuronpedia.api_key("your-api-key-here"):
        # All request classes within this context automatically use the API key
        feature_request = neuronpedia.FeatureRequest()
        model_request = neuronpedia.ModelRequest()
        graph_request = neuronpedia.GraphRequest()

        print("✅ All request classes created with context API key")
        # You can create as many request instances as needed without repeating the key

    # Outside the context, the API key is no longer set
    print("Context exited - API key no longer in scope")


def example_global_setting():
    """Using global context setting (good for application-wide usage)"""
    print("\n=== Global Setting Example ===")

    neuronpedia.set_api_key("your-api-key-here")

    # Now all request classes will use this API key
    feature_request = neuronpedia.FeatureRequest()
    vector_request = neuronpedia.VectorRequest()
    steer_request = neuronpedia.SteerChatRequest()

    print("✅ All request classes created with global context API key")


def example_mixed_usage():
    """Demonstrating mixed usage - explicit parameter overrides context"""
    print("\n=== Mixed Usage Example ===")

    neuronpedia.set_api_key("global-api-key")

    # This uses the global context key
    feature_request1 = neuronpedia.FeatureRequest()

    # This overrides with explicit parameter
    feature_request2 = neuronpedia.FeatureRequest(api_key="explicit-api-key")

    # This uses context manager, temporarily overriding global setting
    with neuronpedia.api_key("context-api-key"):
        feature_request3 = neuronpedia.FeatureRequest()

        # Even explicit parameter still works within context
        feature_request4 = neuronpedia.FeatureRequest(api_key="another-explicit-key")

    print("✅ Mixed usage - explicit parameters override context")


def example_nested_contexts():
    """Demonstrating nested context managers"""
    print("\n=== Nested Contexts Example ===")

    with neuronpedia.api_key("outer-key"):
        feature_request1 = neuronpedia.FeatureRequest()
        print("Outer context: created request with outer-key")

        with neuronpedia.api_key("inner-key"):
            feature_request2 = neuronpedia.FeatureRequest()
            print("Inner context: created request with inner-key")

        # Back to outer context
        feature_request3 = neuronpedia.FeatureRequest()
        print("Back to outer context: created request with outer-key")

    print("✅ Nested contexts work correctly")


if __name__ == "__main__":
    print("Demonstrating context-based API key management...")
    print("Note: Replace 'your-api-key-here' with actual API keys to test")

    # example_context_manager()
    # example_global_setting()
    # example_mixed_usage()
    # example_nested_contexts()

    print("\nThis approach is much cleaner than:")
    print("  feature_request = FeatureRequest(api_key='key')")
    print("  model_request = ModelRequest(api_key='key')")
    print("  graph_request = GraphRequest(api_key='key')")
    print("  # ... repeating for every single request class")
