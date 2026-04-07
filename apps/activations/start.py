import subprocess

from neuronpedia_activations.config import (
    apply_start_config_to_env,
    get_config_from_env,
    parse_start_args,
)

def main() -> None:
    args = parse_start_args()
    apply_start_config_to_env(args)
    runtime_config = get_config_from_env()

    uvicorn_args = [
        "uvicorn",
        "neuronpedia_activations.server:app",
        "--host",
        runtime_config.host,
        "--port",
        str(runtime_config.port),
    ]

    if args.reload:
        uvicorn_args.extend(["--reload"])
        if args.reload_dir:
            uvicorn_args.extend(["--reload-dir", args.reload_dir])

    subprocess.run(uvicorn_args, check=False)


if __name__ == "__main__":
    main()
