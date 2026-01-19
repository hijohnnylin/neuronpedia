#!/usr/bin/env python
"""Simple test script to verify multiprocessing pipeline works."""

import logging
import sys
from pathlib import Path

from chatspace.hf_embed.config import SentenceTransformerConfig
from chatspace.hf_embed.pipeline import run_sentence_transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def test_small_run():
    """Test with a very small dataset."""
    cfg = SentenceTransformerConfig(
        dataset="HuggingFaceTB/smoltalk",
        split="train",
        text_field="messages",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=4,
        rows_per_shard=10,
        max_rows=20,
        output_root=Path("/workspace/test_multiprocessing"),
        dtype="float32",
        device="cpu",
        attention_impl=None,
        compile_model=False,
        progress=True,
        extract_first_assistant=True,
    )

    logging.info("Starting multiprocessing test with config: %s", cfg)

    try:
        result = run_sentence_transformer(cfg)
        logging.info("Test completed successfully!")
        logging.info("Manifest path: %s", result["manifest_path"])
        logging.info("Run path: %s", result["run_path"])
        logging.info("Total rows: %s", result["run_summary"]["rows_total"])
        return True
    except KeyboardInterrupt:
        logging.info("Test interrupted by user (Ctrl-C)")
        return False
    except Exception as exc:
        logging.exception("Test failed with exception: %s", exc)
        return False


if __name__ == "__main__":
    success = test_small_run()
    sys.exit(0 if success else 1)