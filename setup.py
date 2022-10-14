import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='BatchDetect',
    version='0.0.1',
    description='Open source package for detecting batch effect in datasets',
    author='Ali Boushehri',
    author_email='ali.boushehri@roche.com',
    license='MIT',
    keywords='Batch effect, detection',
    url='https://github.com/marrlab/BatchDetect',
    packages=find_packages(exclude=['doc*', 'test*']),
    install_requires=["matplotlib",
                      "numpy",
                      "pandas",
                      "scikit-learn",
                      "seaborn",
                      "umap-learn"],
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
    ],
)
