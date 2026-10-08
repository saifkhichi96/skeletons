from __future__ import annotations

import pytest

from skeletons import (
    SKELETON_ALIASES,
    SUPPORTED_SKELETONS,
    build_layer,
    canonicalize_skeleton_name,
    create,
    get_spec,
    list_supported_skeletons,
)


def test_supported_skeleton_lists_distinguish_canonical_names_and_aliases() -> None:
    alias_names = tuple(SKELETON_ALIASES)

    assert list_supported_skeletons() == SUPPORTED_SKELETONS
    assert list_supported_skeletons(include_aliases=True) == (
        SUPPORTED_SKELETONS + alias_names
    )
    assert "halpe_fullbody" in SUPPORTED_SKELETONS
    assert "halpefullbody" not in SUPPORTED_SKELETONS
    assert SKELETON_ALIASES["halpefullbody"] == "halpe_fullbody"


def test_canonicalize_skeleton_name_resolves_aliases_and_hyphens() -> None:
    assert canonicalize_skeleton_name("HALPEFULLBODY") == "halpe_fullbody"
    assert canonicalize_skeleton_name("halpe-fullbody") == "halpe_fullbody"
    assert canonicalize_skeleton_name("coco-wholebody") == "coco_wholebody"


def test_factory_api_accepts_aliases_and_returns_canonical_specs() -> None:
    spec = get_spec("halpefullbody")
    model = create(
        "halpe-fullbody",
        create_global_orient=False,
        create_body_pose=False,
    )
    layer = build_layer("coco-wholebody")

    assert spec.name == "halpe_fullbody"
    assert model.spec.name == "halpe_fullbody"
    assert layer.spec.name == "coco_wholebody"


def test_unknown_skeleton_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Unknown skeleton name"):
        canonicalize_skeleton_name("unknown-layout")
