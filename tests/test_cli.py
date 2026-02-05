from pathlib import Path

from parsfet.cli import detect_format


def test_detect_format():
    """Verifies file format detection logic.

    Checks:
        - Detection by file extension (.lib, .lef, .techlef, .ict).
        - Priority of forced_format override.
    """
    assert detect_format(Path("foo.lib")) == "lib"
    assert detect_format(Path("foo.lef")) == "lef"
    assert detect_format(Path("foo.techlef")) == "techlef"
    assert detect_format(Path("foo.ict")) == "ict"
    assert detect_format(Path("foo.txt"), forced_format="lib") == "lib"
