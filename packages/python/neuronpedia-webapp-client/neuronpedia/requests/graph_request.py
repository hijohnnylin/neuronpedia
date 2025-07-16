import gzip
import json
import os
from typing import List, Optional

import requests
from neuronpedia.np_graph_metadata import NPGraphMetadata
from neuronpedia.requests.base_request import NPRequest
from requests import Response

# SUPPORTED_GRAPH_MODELS = ["gemma-2-2b", "llama-3.2-1b"]
UPLOAD_FILE_SIZE_LIMIT_BYTES = 100 * 1024 * 1024  # 100MB


class GraphRequest(NPRequest):
    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        super().__init__("graph", api_key=api_key)

    def upload(
        self,
        json_str: str,
    ) -> NPGraphMetadata:
        """
        Upload a JSON graph to Neuronpedia.
        """

        # parse the json string
        loaded_json = json.loads(json_str)

        # TODO: json file checking should be done using the Open Source Circuit Finding repo
        # Check that loaded_json has required fields
        required_fields = ["metadata"]
        for field in required_fields:
            if field not in loaded_json:
                raise ValueError(f"JSON file must contain '{field}' field")

        # Check metadata contains required fields
        if "metadata" in loaded_json:
            required_metadata = [
                "prompt",
                "prompt_tokens",
                "scan",
                "slug",
            ]
            for field in required_metadata:
                if field not in loaded_json["metadata"]:
                    raise ValueError(f"Metadata must contain '{field}' field")

        # we allow any graph to be uploaded
        # if loaded_json["metadata"]["scan"] not in SUPPORTED_GRAPH_MODELS:
        #     raise ValueError(
        #         f"Model {loaded_json['metadata']['scan']} is not supported. Must be one of {SUPPORTED_GRAPH_MODELS}"
        #     )

        model_id = loaded_json["metadata"]["scan"]
        slug = loaded_json["metadata"]["slug"]

        # gzip so we know what the size is
        json_str_gzip = gzip.compress(json_str.encode("utf-8"))

        payload = {
            "filename": slug + ".json",
            "contentLength": len(json_str_gzip),
            "contentType": "application/json",
        }

        response = self.send_request(
            method="POST",
            json=payload,
            uri="signed-put",
        )

        # response will be url and putRequestId
        url = response["url"]
        put_request_id = response["putRequestId"]

        # if either are missing, raise error
        if not url or not put_request_id:
            raise ValueError(
                "Failed to get url or putRequestId from response. Response: {response}"
            )

        print("Upload progress - received signed put request.")

        # upload the file
        response = requests.put(
            url, data=json_str_gzip, headers={"Content-Encoding": "gzip"}
        )

        # check the response is 200
        if response.status_code != 200:
            raise ValueError(f"Failed to upload file to S3. Response: {response.text}")

        print("Upload progress - uploaded file to S3.")

        # now persist it to the database
        response = self.send_request(
            method="POST",
            uri="save-to-db",
            json={"putRequestId": put_request_id},
        )

        print(
            f"Upload to Neuronpedia complete!\nView the graph at the following URL:\n{response['url']}"
        )

        graph_metadata = self.get(model_id, slug)

        return graph_metadata

    def upload_file(
        self,
        filepath: str,
    ) -> NPGraphMetadata:
        """
        Upload a graph file to Neuronpedia.

        Args:
            filepath: Local path to the JSON graph file to upload.

        Returns:
            NPGraphMetadata: The metadata for the uploaded graph. Check its url field to see and use the graph on Neuronpedia, or json_url to get the JSON file.

        Raises:
            FileNotFoundError: If the file doesn't exist at the specified path.
            ValueError: If the file exceeds size limits, isn't valid JSON, or doesn't
                        contain required metadata fields.
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at path: {filepath}")

        if os.path.getsize(filepath) > UPLOAD_FILE_SIZE_LIMIT_BYTES:
            raise ValueError(
                f"File size exceeds the limit of {UPLOAD_FILE_SIZE_LIMIT_BYTES / 1024 / 1024} megabytes"
            )

        # if not json, raise error
        if not filepath.endswith(".json"):
            raise ValueError("File must be a JSON file")

        # load file as string
        with open(filepath, "r") as f:
            json_str = f.read()

        # upload the json
        graph_metadata = self.upload(json_str)

        return graph_metadata

    def delete(self, graph_metadata: NPGraphMetadata) -> Response:
        return self._delete(graph_metadata.model_id, graph_metadata.slug)

    def _delete(self, model_id: str, slug: str) -> Response:
        payload = {
            "modelId": model_id,
            "slug": slug,
        }
        return self.send_request(
            method="POST",
            uri="delete",
            json=payload,
        )

    def get(self, model_id: str, slug: str) -> NPGraphMetadata:
        response = self.send_request(
            method="GET",
            uri=f"{model_id}/{slug}",
        )
        return NPGraphMetadata(
            id=response["id"],
            model_id=response["modelId"],
            slug=response["slug"],
            prompt_tokens=response["promptTokens"],
            prompt=response["prompt"],
            title_prefix=response["titlePrefix"],
            json_url=response["url"],
        )

    def list_owned(self) -> List[NPGraphMetadata]:
        response = self.send_request(
            method="POST",
            uri="list-owned",
        )

        return [
            NPGraphMetadata(
                id=graph["id"],
                model_id=graph["modelId"],
                slug=graph["slug"],
                prompt_tokens=graph["promptTokens"],
                prompt=graph["prompt"],
                title_prefix=graph["titlePrefix"],
                json_url=graph["url"],
            )
            for graph in response
        ]

    def generate(
        self,
        model_id: str,
        prompt: str,
        slug: str,
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        node_threshold: float = 0.8,
        edge_threshold: float = 0.98,
    ) -> NPGraphMetadata:
        payload = {
            "modelId": model_id,
            "prompt": prompt,
            "slug": slug,
            "maxNLogits": max_n_logits,
            "desiredLogitProb": desired_logit_prob,
            "nodeThreshold": node_threshold,
            "edgeThreshold": edge_threshold,
        }
        self.send_request(
            method="POST",
            uri="generate",
            json=payload,
        )

        # now get the graph metadata
        graph_metadata = self.get(model_id, slug)
        return graph_metadata
