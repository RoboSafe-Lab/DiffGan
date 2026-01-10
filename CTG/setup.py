from setuptools import setup, find_packages


setup(
    name="tbsim",
    packages=[package for package in find_packages() if package.startswith("tbsim")],
    install_requires=[
        "l5kit==1.5.0",
        "numpy~=1.19.5",  # need to manually update numpy version to (1.19.5)
        "numba== 0.56.4",
        "pytorch-lightning==1.9.4",
        "wandb",
        "torch==1.11",
        "torchvision==0.12.0",
        "pyemd",
        "h5py",
        "imageio-ffmpeg",
        "casadi",
        "protobuf==3.20.1", # new version might cause error
        "einops==0.6.0",
        "torchtext",  # weird pytorch-lightning dependency bug
        "opencv-python~=4.5.5.64",
        "scipy~=1.10.1",
        "seaborn~=0.12.0"

    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Traffic Behavior Simulation",
    author="NVIDIA AV Research",
    author_email="danfeix@nvidia.com",
    version="0.0.1",
)
