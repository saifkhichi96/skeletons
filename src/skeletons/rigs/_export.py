from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from xml.dom import minidom

from ._spec import SkeletonSpec


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _write(root: ET.Element, path: str | Path) -> None:
    text = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(
        indent="  "
    )
    Path(path).write_text(text, encoding="utf-8")


def _axis(spec: SkeletonSpec, index: int) -> tuple[float, ...]:
    axis = (spec.joint_axes or (None,) * spec.num_joints)[index] or (1.0, 0.0, 0.0)
    norm = sum(value * value for value in axis) ** 0.5
    return tuple(value / norm for value in axis)


def _validate_limits(spec: SkeletonSpec) -> None:
    for index in range(spec.num_joints):
        limits = (spec.joint_limits or (None,) * spec.num_joints)[index]
        if spec.joint_type(index) in {"ball", "free"} and limits is not None:
            raise ValueError(
                "Axis-angle component limits cannot be faithfully exported for ball/free joints."
            )


def export_urdf(spec: SkeletonSpec, path: str | Path) -> None:
    """Export any rig as a URDF kinematic tree.

    Parameters
    ----------
    spec : SkeletonSpec
        Rig topology and optional physical metadata.
    path : str or Path
        Destination file. Ball joints expand to three continuous Euler axes;
        exported joint coordinates are not the model's axis-angle q values.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ball limits or reserved generated link names cannot be represented.
    OSError
        If the destination cannot be written.
    """
    _validate_limits(spec)
    robot = ET.Element("robot", name=spec.name)
    world = "__world"
    if world in spec.joint_names:
        raise ValueError("The link name __world is reserved for URDF export.")
    ET.SubElement(robot, "link", name=world)
    links = {link.name: link for link in spec.links}
    for name in spec.joint_names:
        element = ET.SubElement(robot, "link", name=name)
        if name in links:
            link = links[name]
            inertial = ET.SubElement(element, "inertial")
            ET.SubElement(inertial, "origin", xyz=_fmt(link.com), rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=str(link.mass))
            ixx, iyy, izz = link.inertia_diag
            ET.SubElement(
                inertial,
                "inertia",
                ixx=str(ixx),
                iyy=str(iyy),
                izz=str(izz),
                ixy="0",
                ixz="0",
                iyz="0",
            )
        geometry = ET.SubElement(ET.SubElement(element, "visual"), "geometry")
        ET.SubElement(geometry, "sphere", radius="0.035")

    def joint(
        name: str, parent: str, child: str, kind: str, offset, axis=None, limits=None
    ) -> None:
        element = ET.SubElement(robot, "joint", name=name, type=kind)
        ET.SubElement(element, "parent", link=parent)
        ET.SubElement(element, "child", link=child)
        ET.SubElement(element, "origin", xyz=_fmt(offset), rpy="0 0 0")
        if axis is not None:
            ET.SubElement(element, "axis", xyz=_fmt(axis))
        if kind == "revolute":
            ET.SubElement(
                element,
                "limit",
                lower=str(limits[0]),
                upper=str(limits[1]),
                effort="100",
                velocity="10",
            )

    for index in range(spec.num_joints):
        parent_index = spec.parents[index]
        parent = world if parent_index < 0 else spec.joint_names[parent_index]
        child = spec.joint_names[index]
        offset = (
            (0, 0, 0) if index == spec.root_index else spec.rest_offsets[index].tolist()
        )
        kind = spec.joint_type(index)
        limits = (spec.joint_limits or (None,) * spec.num_joints)[index]
        if kind == "ball":
            intermediate = [f"{child}__ball_x", f"{child}__ball_y"]
            if any(name in spec.joint_names for name in intermediate):
                raise ValueError("Rig names collide with generated ball-joint links.")
            for name in intermediate:
                ET.SubElement(robot, "link", name=name)
            chain = [parent, *intermediate, child]
            for axis_index, axis in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                joint(
                    f"{child}__axis_{axis_index}",
                    chain[axis_index],
                    chain[axis_index + 1],
                    "continuous",
                    offset if axis_index == 0 else (0, 0, 0),
                    axis,
                )
        elif kind == "hinge":
            joint(
                f"{child}__joint",
                parent,
                child,
                "revolute" if limits else "continuous",
                offset,
                _axis(spec, index),
                limits[0] if limits else None,
            )
        else:
            joint(
                f"{child}__joint",
                parent,
                child,
                "floating" if kind == "free" else "fixed",
                offset,
            )
    generated_names = set(spec.joint_names) | {world}
    generated_names.update(
        f"{name}__ball_{axis}" for name in spec.joint_names for axis in ("x", "y")
    )
    for kind, frames in (("marker", spec.markers), ("contact", spec.contacts)):
        for frame in frames:
            name = f"{kind}__{frame.name}"
            if name in generated_names:
                raise ValueError(
                    "Rig names collide with generated attached-frame links."
                )
            generated_names.add(name)
            ET.SubElement(robot, "link", name=name)
            joint(name + "__joint", frame.parent, name, "fixed", frame.local_xyz)
    _write(robot, path)


def export_mjcf(spec: SkeletonSpec, path: str | Path) -> None:
    """Export any rig to MJCF with native free, ball, hinge, and fixed bodies.

    Parameters
    ----------
    spec : SkeletonSpec
        Rig definition. Explicit links supply mass, COM, and diagonal inertia.
    path : str or Path
        Destination file. Spheres and unit masses are illustrative defaults
        when physical metadata is absent.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If component-wise ball limits cannot be represented.
    OSError
        If the destination cannot be written.
    """
    _validate_limits(spec)
    root = ET.Element("mujoco", model=spec.name)
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    links = {link.name: link for link in spec.links}
    children = {index: [] for index in range(spec.num_joints)}
    for child, parent in enumerate(spec.parents):
        if parent >= 0:
            children[parent].append(child)

    def body(parent: ET.Element, index: int) -> None:
        name = spec.joint_names[index]
        offset = (
            (0, 0, 0) if index == spec.root_index else spec.rest_offsets[index].tolist()
        )
        element = ET.SubElement(parent, "body", name=name, pos=_fmt(offset))
        kind = spec.joint_type(index)
        if kind == "free":
            if index != spec.root_index:
                raise ValueError("MJCF free joints are supported only at the root.")
            ET.SubElement(element, "freejoint", name=name + "__joint")
        elif kind == "ball":
            ET.SubElement(element, "joint", name=name + "__joint", type="ball")
        elif kind == "hinge":
            limits = (spec.joint_limits or (None,) * spec.num_joints)[index]
            attributes = (
                {"range": _fmt(limits[0]), "limited": "true"}
                if limits
                else {"limited": "false"}
            )
            ET.SubElement(
                element,
                "joint",
                name=name + "__joint",
                type="hinge",
                axis=_fmt(_axis(spec, index)),
                **attributes,
            )
        if name in links:
            link = links[name]
            ET.SubElement(
                element,
                "inertial",
                pos=_fmt(link.com),
                mass=str(link.mass),
                diaginertia=_fmt(link.inertia_diag),
            )
        ET.SubElement(element, "geom", type="sphere", size="0.035", mass="1")
        for kind, frames in (("marker", spec.markers), ("contact", spec.contacts)):
            for frame in frames:
                if frame.parent == name:
                    ET.SubElement(
                        element,
                        "site",
                        name=f"{kind}__{frame.name}",
                        pos=_fmt(frame.local_xyz),
                        size=str(getattr(frame, "radius", 0.01)),
                    )
        for child in children[index]:
            body(element, child)

    body(world, spec.root_index)
    _write(root, path)
