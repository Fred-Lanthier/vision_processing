"""Template generique de collecte de donnees ManiSkill3.

Modules :
  urdf_compose   : fusion robot + outil en un seul URDF (approche B).
  robot_spec     : RobotSpec/ToolSpec/CameraSpec + build_agent dynamique.
  scene_config   : ObjectConfig / RandomizationConfig + exemples.
  env_template   : GenericCollectEnv (load_scene + domain randomization).
  policy_base    : ScriptedPolicy (interface) + ReachHoldPolicy (exemple).
  data_collector : TrajectoryRecorder (buffers RAM -> flush disque).
  collect        : point d'entree assemblant le tout.
"""
