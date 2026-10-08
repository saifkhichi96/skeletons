from __future__ import annotations

import pytest
from playground.rom import (
    AxisRomLimit,
    apply_rom_payload,
    default_rom_limits,
    serialize_rom_limits,
)


def test_default_rom_limits_create_disabled_axes() -> None:
    limits = default_rom_limits(["root", "knee"])

    assert set(limits) == {"root", "knee"}
    assert len(limits["root"]) == 3
    assert all(not limit.enabled for limit in limits["root"])
    assert limits["root"][0] is not limits["root"][1]


def test_rom_payload_round_trips_enabled_limits() -> None:
    limits = default_rom_limits(["root", "knee"])
    limits["knee"][0] = AxisRomLimit(
        enabled=True,
        minimum_deg=-45.0,
        maximum_deg=10.0,
    )

    payload = serialize_rom_limits(limits, skeleton="human36m")
    parsed = apply_rom_payload(
        payload,
        skeleton="human36m",
        joint_names=["root", "knee"],
    )

    assert payload["skeleton"] == "human36m"
    assert "root" not in payload["joint_limits"]
    assert parsed["knee"][0].enabled is True
    assert parsed["knee"][0].minimum_deg == -45.0
    assert parsed["knee"][0].maximum_deg == 10.0
    assert parsed["knee"][1].enabled is False


def test_apply_rom_payload_rejects_wrong_skeleton_and_bad_schema() -> None:
    with pytest.raises(ValueError, match="does not match"):
        apply_rom_payload(
            {"skeleton": "coco", "joint_limits": {}},
            skeleton="human36m",
            joint_names=["root"],
        )

    with pytest.raises(ValueError, match="joint_limits"):
        apply_rom_payload(
            {"joint_limits": []},
            skeleton="human36m",
            joint_names=["root"],
        )

    with pytest.raises(ValueError, match="min greater than max"):
        apply_rom_payload(
            {
                "joint_limits": {
                    "root": {"x": {"enabled": True, "min_deg": 30.0, "max_deg": 10.0}}
                }
            },
            skeleton="human36m",
            joint_names=["root"],
        )


def test_apply_rom_payload_ignores_unknown_joints() -> None:
    parsed = apply_rom_payload(
        {
            "joint_limits": {
                "unknown": {"x": {"enabled": True, "min_deg": -1.0, "max_deg": 1.0}},
                "root": {"z": {"enabled": True, "min_deg": -2.0, "max_deg": 2.0}},
            }
        },
        skeleton="human36m",
        joint_names=["root"],
    )

    assert set(parsed) == {"root"}
    assert parsed["root"][0].enabled is False
    assert parsed["root"][2].enabled is True
    assert parsed["root"][2].minimum_deg == -2.0
