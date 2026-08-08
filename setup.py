from setuptools import find_packages, setup

setup(
    name="rsl_rl",
    version="0.1.0",
    description="Custom rsl_rl fork (my_rsl_rl) installed under the top-level name 'rsl_rl'.",
    packages=["rsl_rl"] + ["rsl_rl." + p for p in find_packages(where=".")],
    package_dir={"rsl_rl": "."},
    install_requires=[
        "torch",
        "numpy",
        "gymnasium",
        "tensorboard",
        "GitPython",
        "wandb",
    ],
)
