"""Main script calls"""

import subprocess

import pytest


def assert_proc_ended_okay(proc):
    """Actually just hard-coded checks for logging output"""

    # gather last string in output
    out = []
    for s in proc.stderr:
        out.append(s)

    # must basically be this:
    assert "INFO:root:Total reward:" in str(out[-1]), str(out)


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
    assert_proc_ended_okay(proc)


def test_flat_po_zero():
    """Tests po-zero in :file:`flat_pomdp_experiment.py`"""

    for model_input in ["state", "history"]:

        for model_output in ["value_and_prior", "q_values"]:

            proc = subprocess.Popen(
                [
                    "python",
                    "flat_pomdp_experiment.py",
                    "conf/flat_pomdp/tiger.pomdp",
                    "po-zero",
                    "conf/solutions/po_zero_example.yaml",
                    f"model_input={model_input}",
                    f"model_output={model_output}"
                ],
                stderr=subprocess.PIPE,
            )
            assert_proc_ended_okay(proc)


if __name__ == "__main__":
    pytest.main([__file__])
