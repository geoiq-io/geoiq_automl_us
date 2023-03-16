#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Balveer Singh",
    author_email='balveer@geoiq.io',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Get started with a wide range of location-based features and build ml models using this package",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='geoiq_automl_us',
    name='geoiq_automl_us',
    packages=find_packages(include=['geoiq_automl_us', 'geoiq_automl_us.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/geoiq-io/geoiq_automl_us',
    version='0.1.0',
    zip_safe=False,
)
