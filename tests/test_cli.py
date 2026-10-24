"""Tests for CLI arguments and subcommands."""

import os
import tempfile
from driftpulse.cli.main import build_parser, main


def test_cli_parser_help():
    parser = build_parser()
    assert parser.prog == "driftpulse"


def test_cli_quicktest():
    ret = main(["quicktest"])
    assert ret == 0


def test_cli_profile_and_compare():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_ref = os.path.join(tmpdir, "ref.csv")
        csv_curr = os.path.join(tmpdir, "curr.csv")
        profile_json = os.path.join(tmpdir, "baseline.json")
        html_out = os.path.join(tmpdir, "report.html")

        # Create sample CSVs
        with open(csv_ref, "w", encoding="utf-8") as f:
            f.write("val,cat\n" + "\n".join([f"{i},A" for i in range(100)]))

        with open(csv_curr, "w", encoding="utf-8") as f:
            f.write("val,cat\n" + "\n".join([f"{i+1},A" for i in range(100)]))

        # Test profile command
        ret_prof = main(["profile", csv_ref, "--out", profile_json])
        assert ret_prof == 0
        assert os.path.exists(profile_json)

        # Test compare command
        ret_comp = main(["compare", csv_ref, csv_curr, "--html", html_out])
        assert ret_comp == 0
        assert os.path.exists(html_out)
