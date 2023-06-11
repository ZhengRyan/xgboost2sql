#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

NAME = 'xgboost2sql'



def get_requirements(stage=None):
    file_name = 'requirements'

    if stage is not None:
        file_name = f"{file_name}-{stage}"

    requirements = []
    with open(f"{file_name}.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('-'):
                continue

            requirements.append(line)

    return requirements

test_requirements = ['pytest>=3', ]

setup(
    author="RyanZheng",
    author_email='zhengruiping000@163.com',
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
    description="Convert the trained xgboost model to sql",
    entry_points={
        'console_scripts': [
            'xgboost2sql=xgboost2sql.cli:main',
        ],
    },
    install_requires=get_requirements(),
    license="MIT license",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='xgboost2sql',
    name=NAME,
    packages=find_packages(include=['xgboost2sql', 'xgboost2sql.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ZhengRyan/xgboost2sql',
    version='0.1.0',
    zip_safe=False,
)
