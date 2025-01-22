from setuptools import setup, find_packages

setup(
    name="sign_language_translation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "pyyaml",
        "tqdm"
    ],
    entry_points={
        'console_scripts': [
            'train-sign-language=train:main',
        ],
    },
    author="Your Name",
    description="A package for sign language translation using deep learning models in PyTorch.",
    url="https://github.com/yourusername/sign_language_translation"
)
