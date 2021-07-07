from setuptools import setup, find_packages
import sys

# setup nltk
import nltk
nltk.download('averaged_perceptron_tagger')

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python >=3.6 is required for AutoDiscourse.')

with open('README.md', encoding="utf8") as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

if __name__ == '__main__':
    setup(
        name='usl_score',
        version='0.1',
        author="Vitou Phy",
        author_email="phy_vitou@live.com",
        description='Hierarchical Evaluation Metric for Dialogue (USL-H)',
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/vitouphy/usl_dialogue_metric",
        license=license,
        python_requires='>=3.6',
        packages=find_packages(
            exclude=('logs', 'tests')
        ),
        entry_points={
            'console_scripts': [
                "usl_score=usl_score.score:compute"
            ]
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent"
        ],
        install_requires=reqs.strip().split('\n'),
        # include_package_data=True,
        # test_suite='tests.suites.unittests',
    )
