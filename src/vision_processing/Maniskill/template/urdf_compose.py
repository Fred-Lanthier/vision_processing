"""
urdf_compose.py
================
Fusion de deux URDF (robot + outil) en UN seul fichier URDF (approche B).

Pourquoi ? Dans ManiSkill3, un agent ne charge qu'UN seul `urdf_path` -> une
seule articulation. Pour que les liens de l'outil (ex: `fork_tip`) apparaissent
dans `robot.links_map`, que la cinematique avant (FK) soit coherente et que
`ee_link_name` / le tracking de pose fonctionnent uniformement, on injecte
l'URDF de l'outil comme un joint FIXE enfant du lien parent du robot, puis on
ecrit un URDF fusionne temporaire que l'on passe a ManiSkill.

Tous les chemins de meshes (robot ET outil) sont resolus en chemins ABSOLUS
dans le fichier fusionne, car ce dernier est ecrit dans un repertoire temporaire
ou les chemins relatifs d'origine ne sont plus valides.
"""

import os
import tempfile
import xml.etree.ElementTree as ET


def _resolve_mesh_paths(robot_elem: ET.Element, base_dir: str) -> None:
    """Rend absolus tous les <mesh filename="..."> relatifs, en place.

    Les chemins `package://` sont laisses tels quels (resolus par ROS/ManiSkill).
    """
    for mesh in robot_elem.iter("mesh"):
        fn = mesh.get("filename")
        if fn is None:
            continue
        if fn.startswith("package://") or os.path.isabs(fn):
            continue
        mesh.set("filename", os.path.abspath(os.path.join(base_dir, fn)))


def _find_root_link(robot_elem: ET.Element) -> str:
    """Retourne le nom du lien racine (lien jamais 'child' d'un joint)."""
    all_links = {l.get("name") for l in robot_elem.findall("link")}
    child_links = {
        j.find("child").get("link")
        for j in robot_elem.findall("joint")
        if j.find("child") is not None
    }
    roots = all_links - child_links
    if not roots:
        raise ValueError("Aucun lien racine trouve dans l'URDF de l'outil.")
    # Si plusieurs racines, on prend la premiere de maniere deterministe.
    return sorted(roots)[0]


def compose_robot_tool(
    robot_urdf_path: str,
    tool_urdf_path: str,
    parent_link: str,
    attach_xyz=(0.0, 0.0, 0.0),
    attach_rpy=(0.0, 0.0, 0.0),
    out_path: str | None = None,
) -> str:
    """Fusionne `tool_urdf_path` dans `robot_urdf_path` via un joint fixe.

    Args:
        robot_urdf_path: URDF du robot seul.
        tool_urdf_path:  URDF de l'outil seul.
        parent_link:     lien du robot auquel attacher l'outil (ex: "panda_hand").
        attach_xyz:      translation (m) du joint fixe parent -> racine outil.
        attach_rpy:      rotation (rad, roll-pitch-yaw) du joint fixe.
        out_path:        chemin de sortie. Si None, un fichier temporaire est cree.

    Returns:
        Le chemin du fichier URDF fusionne.
    """
    robot_dir = os.path.dirname(os.path.abspath(robot_urdf_path))
    tool_dir = os.path.dirname(os.path.abspath(tool_urdf_path))

    robot_tree = ET.parse(robot_urdf_path)
    robot_root = robot_tree.getroot()
    tool_root = ET.parse(tool_urdf_path).getroot()

    # 1) Resoudre les meshes en absolu (le fichier fusionne vit ailleurs).
    _resolve_mesh_paths(robot_root, robot_dir)
    _resolve_mesh_paths(tool_root, tool_dir)

    # 2) Verifier que le lien parent existe bien cote robot.
    robot_link_names = {l.get("name") for l in robot_root.findall("link")}
    if parent_link not in robot_link_names:
        raise ValueError(
            f"parent_link '{parent_link}' introuvable dans l'URDF du robot. "
            f"Liens disponibles: {sorted(robot_link_names)}"
        )

    tool_root_link = _find_root_link(tool_root)

    # 3) Detecter d'eventuelles collisions de noms (informationnel).
    robot_joint_names = {j.get("name") for j in robot_root.findall("joint")}
    for tool_link in tool_root.findall("link"):
        if tool_link.get("name") in robot_link_names:
            raise ValueError(
                f"Collision de nom de lien: '{tool_link.get('name')}' existe deja "
                f"dans le robot. Renommez les liens de l'outil."
            )

    # 4) Copier liens + joints de l'outil dans le robot.
    for tool_link in tool_root.findall("link"):
        robot_root.append(tool_link)
    for tool_joint in tool_root.findall("joint"):
        if tool_joint.get("name") in robot_joint_names:
            tool_joint.set("name", f"tool_{tool_joint.get('name')}")
        robot_root.append(tool_joint)

    # 5) Creer le joint fixe d'attache parent_link -> racine outil.
    attach = ET.SubElement(robot_root, "joint")
    attach.set("name", "tool_attach_joint")
    attach.set("type", "fixed")
    ET.SubElement(attach, "parent").set("link", parent_link)
    ET.SubElement(attach, "child").set("link", tool_root_link)
    origin = ET.SubElement(attach, "origin")
    origin.set("xyz", " ".join(str(v) for v in attach_xyz))
    origin.set("rpy", " ".join(str(v) for v in attach_rpy))

    # 6) Ecrire le fichier fusionne.
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix="_merged.urdf", prefix="ms3_")
        os.close(fd)
    robot_tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path
