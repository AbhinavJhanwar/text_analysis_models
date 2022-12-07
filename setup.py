from setuptools import setup, find_packages

base_packages = [
        "scikit-learn>=0.22.2",
        "numpy>=1.18.5",
        "pandas>=1.3.5",
        "nltk>=3.6.7",
        "plotly>=5.5.0",
        "bs4",
        "tqdm>=4.62.3",
        "spacy>=3.2.3",
        "pyarrow>=6.0.1",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0.tar.gz"
]

setup(
    name='text-analysis-models',
    packages=find_packages(exclude=["notebooks", "docs"]),
    version='0.1.0',    
    description='A text semantics api for text keyword extraction and insights, sentiment analysis and topic modeling',
    url='https://github.com/AbhinavJhanwar/text-analysis-models.git',
    author='Abhinav Jhanwar',
    author_email='abhij.1994@gmail.com',
    install_requires=base_packages,
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)