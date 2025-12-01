from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Cette fonction va lire ton package.xml et extraire les infos
d = generate_distutils_setup(
    packages=['vision_processing', 'diffusion_model', 'calories_app', 'vision_segmentation', 'vision_tracking'],  # Le nom exact du dossier dans src/
    package_dir={'': 'src'}          # On dit à catkin que le code est dans src
)

setup(**d)