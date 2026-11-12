from agent_loop.cli import main


def test_cli_runs_without_a_subcommand() -> None:
    assert main([]) == 0
