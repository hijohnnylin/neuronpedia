def main():
    # Lazy import to avoid import cost when used as a library
    from chatspace.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
