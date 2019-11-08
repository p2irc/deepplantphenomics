from setuptools import setup

setup(
    name='deepplantphenomics',
    version='',
    packages=['deepplantphenomics'],
    package_data={'deepplantphenomics': ['network_states/*', 'network_states/**/*']},
    url='',
    license='MIT',
    author='Jordan Ubbens',
    author_email='jordan.ubbens@usask.ca',
    description='Deep learning tools for plant phenotyping',
    install_requires=[
        'tensorflow<=1.15',
        'numpy',
        'tqdm',
        'opencv-python',
        'scipy',
        'pillow'
    ]
)
