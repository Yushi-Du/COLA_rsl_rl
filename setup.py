from setuptools import find_packages, setup

setup(
    name="rsl_rl",
    version="0.1.0",
    description="COLA's RSL-RL fork installed under the package name 'rsl_rl'.",
    license="MIT",
    classifiers=["License :: OSI Approved :: MIT License"],
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
