from leann.cli import LeannCLI


def test_cli_ask_accepts_positional_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["ask", "my-docs", "Where are prompts configured?"])

    assert args.command == "ask"
    assert args.index_name == "my-docs"
    assert args.query == "Where are prompts configured?"
