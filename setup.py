from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Read the requirements from a file and return as a list."""
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements




setup(    name='ml_project',
    version='0.1.0',
    author='Harsh',
    author_email='cse24046@iiitkalyani.ac.in',
    description='A machine learning project setup',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.6',

)