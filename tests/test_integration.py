"""Main script calls"""

import subprocess

import pytest


def proc_ended_okay(proc):
    """Actually just hard-coded checks for logging output"""

    # gather last string in output
    s = ""
    for s in proc.stderr:
        pass

    # must basically be this:
    return "INFO:root:Total reward:" in str(s)


def test_flat_pouct():
    """Tests po-uct in :file:`flat_pomdp_experiment.py`"""

    proc = subprocess.Popen(
        [
            "python",
            "flat_pomdp_experiment.py",
            "conf/flat_pomdp/tiger.pomdp",
            "po-uct",
            "conf/solutions/pouct_example.yaml",
        ],
        stderr=subprocess.PIPE,
    )
    assert proc_ended_okay(proc)


def test_flat_po_zero():
    """Tests po-zero in :file:`flat_pomdp_experiment.py`"""

    for solution in ["po-zero-state", "po-zero-history"]:

        proc = subprocess.Popen(
            [
                "python",
                "flat_pomdp_experiment.py",
                "conf/flat_pomdp/tiger.pomdp",
                solution,
                "conf/solutions/po_zero_example.yaml",
            ],
            stderr=subprocess.PIPE,
        )
        assert proc_ended_okay(proc)


if __name__ == "__main__":
    pytest.main([__file__])
