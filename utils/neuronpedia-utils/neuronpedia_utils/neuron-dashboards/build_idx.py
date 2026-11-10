"""Write neurons/layer_XX.idx next to each layer_XX.index.json in R2.

The .idx file holds the same N + 1 offsets as little-endian uint64. Neuron n needs only bytes
8n .. 8n+15 of it, so a browser gets its byte range with one 16-byte Range request.

Usage: python build_idx.py MODEL [MODEL ...]
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np

BUCKET = "neuronpedia"
PREFIX = "neuron-dashboards/v2"


def client():
    """Return an S3 client for R2."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_S3_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def convert(model, layer):
    """Convert one layer's index; return (layer, neuron count, last offset)."""
    s3 = client()
    base = f"{PREFIX}/{model}/neurons/layer_{layer:02d}"
    d = json.loads(
        s3.get_object(Bucket=BUCKET, Key=f"{base}.index.json")["Body"].read()
    )
    offs = np.asarray(d["offsets"], dtype="<u8")
    assert (
        d["layer"] == layer
        and offs[0] == 0
        and np.all(np.diff(offs.astype(np.int64)) > 0)
    )
    size = s3.head_object(Bucket=BUCKET, Key=f"{base}.json.gz")["ContentLength"]
    assert int(offs[-1]) == size, (layer, int(offs[-1]), size)
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{base}.idx",
        Body=offs.tobytes(),
        ContentType="application/octet-stream",
    )
    back = s3.get_object(Bucket=BUCKET, Key=f"{base}.idx")["Body"].read()
    assert back == offs.tobytes()
    return layer, len(offs) - 1, int(offs[-1])


def main():
    """Convert every layer of each model named on the command line."""
    for model in sys.argv[1:]:
        s3 = client()
        meta = json.loads(
            s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{model}/meta.json")[
                "Body"
            ].read()
        )
        n_layers, n_neurons = meta["layers"], meta["neurons_per_layer"]
        with ThreadPoolExecutor(16) as ex:
            for layer, n, size in ex.map(
                lambda li: convert(model, li), range(n_layers)
            ):
                assert n == n_neurons, (model, layer, n)
                print(
                    f"{model} layer {layer:02d}: {n} neurons, {size / 2**20:.0f} MiB",
                    flush=True,
                )
        print(f"{model}: {n_layers} .idx files done", flush=True)


if __name__ == "__main__":
    main()
