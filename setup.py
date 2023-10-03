from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Sensor fusion algorithms'
LONG_DESCRIPTION = 'An educational repo to teach people robotics principels'

# Setting up
setup(
        name="sensor_fusion", 
        version=VERSION,
        author="Benjamin Evans",
        author_email="<bdevans@sun.ac.za>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'autonomous racing'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux",
        ]
)
