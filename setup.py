import os
from setuptools import find_packages, setup


__version__ = "0.1"

if "VERSION" in os.environ:
    BUILD_NUMBER = os.environ["VERSION"].rsplit(".", 1)[-1]
else:
    BUILD_NUMBER = os.environ.get("BUILD_NUMBER", "dev")

dependencies = [
    "click>=7.0",
    "numpy>=1.17.0,<2.0",
    "torch==1.5.0",
    "torchvision==0.6.0",
    "opencv-python>=4.1.1.26,<5.0",
    "tensorboard==2.3.0",
    "scikit-learn>=0.21.0,<1.0",
    "scikit-image",
    "git-python>=1.0.3",
    "gin-config==0.3.0",
    "torchsummary>=1.5.1",
    "xlsxwriter>=1.2.9",
    "nibabel",
    "nilearn",
    "black",
]

setup(
    name="visualizer",
    version="{0}.{1}".format(__version__, BUILD_NUMBER),
    description="Visualizing CNNs for Semantic Segmentation of BraTS 2019 Dataset",
    author="Deepan Chakravarthi Padmanabhan",
    install_requires=dependencies,
    packages=find_packages(),
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            "visualizer_train=visualizer.train:train",
            "visualizer_evaluate=visualizer.evaluate.evaluate:evaluate",
        ]
    ),
    data_files=[
        (
            "visualizer_train_config",
            [
                "config/train.gin",
                "config/evaluate.gin",
            ],
        )
    ],
    python_requires=">=3.6.9,<3.9",
)
