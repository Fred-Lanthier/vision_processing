# Template générique de collecte de données — ManiSkill3

Squelette réutilisable, extrait de `Simulation_multiple_save_data.py`, pour
collecter des datasets de démonstrations expertes dans **n'importe quel**
environnement ManiSkill3 (pas seulement la robotique d'assistance).

L'idée : tout ce qui est **infrastructure** (robot, capteurs, randomisation,
enregistrement) est factorisé et piloté par config ; seule la **politique
experte** reste spécifique à la tâche.

## Fichiers

| Fichier | Rôle | À modifier ? |
|---|---|---|
| `urdf_compose.py` | Fusionne robot + outil en 1 URDF (approche B) | Non (utilitaire) |
| `robot_spec.py` | `RobotSpec`/`ToolSpec`/`CameraSpec` + `build_agent` | Non (utilitaire) |
| `data_collector.py` | `TrajectoryRecorder` (RAM → disque) | Non (utilitaire) |
| `env_template.py` | `GenericCollectEnv` (scène + domain randomization) | Parfois (scène) |
| `scene_config.py` | `ObjectConfig` / `RandomizationConfig` | **Oui** (tes objets) |
| `policy_base.py` | Interface `ScriptedPolicy` + exemple | **Oui** (ta politique) |
| `cam_utils.py` | `dir_to_quat_wxyz` / `look_at_quat_wxyz` (orientation caméra SAPIEN) | Non (utilitaire) |
| `collect.py` | Assemble tout + boucle de collecte | **Oui** (point d'entrée) |

### Exemple concret : pick-and-place sur étagère

| Fichier | Rôle |
|---|---|
| `env_pick_place.py` | `PickPlaceShelfEnv` : cube dynamique (préhensible) + étagère fixe à droite |
| `policy_pick_place.py` | `PickPlacePolicy` : FSM 8 phases (hover→descend→grasp→lift→transport→place→release→retreat) |
| `collect_pick_place.py` | Panda standard + 2 caméras (poignet + statique-gauche 45°) + recorder |

```bash
source venv_sam3/bin/activate
cd src/vision_processing/Maniskill/template
python3 collect_pick_place.py
```

Points illustrés par cet exemple :
- **Gripper standard** (`tool=None`, Panda nu) pour une préhension physique réelle.
- **Caméra fixe dans le monde** montée sur la base `panda_link0` (lien fixe) avec
  une pose décalée + orientée par `look_at_quat_wxyz`.
- **Grasp top-down sans contrôle d'orientation** : la politique ne commande que
  position + gripper ; le homing oriente le gripper vers le bas et
  `pd_ee_delta_pose` maintient cette orientation.

## Principe de design

1. **L'URDF est la source de vérité** pour la cinématique et le montage des
   capteurs. Ce que l'URDF ne peut pas porter dans ManiSkill (mode de contrôle,
   keyframe de homing, lien EE contrôlé, liens enregistrés) vit dans une petite
   `RobotSpec`.
2. **Caméras listées explicitement** dans la spec (`CameraSpec`), montées sur un
   lien nommé.
3. **Liens enregistrés configurables** (`record_links`) : remplace le `fork_tip`
   codé en dur. On enregistre position + quaternion (rel. base) de chaque lien.
4. **Outil dans un URDF séparé** (`ToolSpec`) : fusionné automatiquement dans
   l'URDF du robot via un joint fixe (`parent_link`, `attach_xyz/rpy`) → une
   seule articulation, donc `fork_tip` apparaît dans `links_map` et le tracking
   de pose marche uniformément. Si l'outil est déjà dans l'URDF, mettre
   `tool=None`.

## Lancer

```bash
source venv_sam3/bin/activate
cd src/vision_processing/Maniskill/template
python3 collect.py
```

> Les scripts se lancent **directement** (`python3 <fichier>.py`) depuis le
> dossier `template/`. Ils utilisent des imports de modules voisins, ce qui
> contourne le `vision_processing/__init__.py` (qui importe `rospy`, absent du
> venv). Lancer via `-m vision_processing...` échouerait pour cette raison.

Sortie : `datas/Template_record_TEST/Trajectory_<id>/` avec le JSON d'états +
images RGB/depth des caméras (même format que le pipeline original).

## Adapter à une nouvelle tâche

1. **Objets** → édite `scene_config.py` (`ObjectConfig`) avec tes meshes/primitives.
2. **Robot/outil/caméras** → édite `make_robot_spec()` dans `collect.py`.
   - Outil séparé : remplis `ToolSpec` (Exemple B commenté).
3. **Politique** → copie `ReachHoldPolicy` dans `policy_base.py` et écris ta FSM
   vectorisée (phases approach → grasp → transport → place…), en tenseurs `(B,)`.
4. **Échelle** → augmente `num_envs` dans `collect.py` (gratuit, vectorisé GPU).

## Limites connues / à valider

- L'auto-détection du lien de caméra dans `data_collector.py` suppose une
  convention de nommage (`camera_<uid>_link`, `<uid>_link`, `camera_<uid>`). Si
  tes liens ne suivent pas ce schéma, passe un mapping explicite.
- `GenericCollectEnv` sous-classe `PickCubeEnv` (cube par défaut caché) pour
  hériter de la table/lumière éprouvées. Pour une scène radicalement différente,
  remplace la classe de base par `BaseEnv` + `TableSceneBuilder`.
- **Non exécuté end-to-end** ici (nécessite GPU + assets). Les modules purs
  (`urdf_compose`) sont testés ; la boucle complète reste à valider sur ta
  machine.
