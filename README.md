# Cascade Facial Emotion Recognition
Facial Emotion Recognition with
* Cascade method to detect faces
* DNN-based method to classify facial emotions

In the current implementation, whenever a face is detected in a frame, the CNN model will classify the person's emotion. This emotion
is then accumulated over a fixed `window_duration_in_frames` frames to output the dominant emotion using a Queue. In case no face is detected, this Queue remains untouched.

# Prerequisites
- [x] poetry
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source ~/.profile
poetry
```
- [x] conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
rm Anaconda3-2021.05-Linux-x86_64.sh
```
- [x] CUDA == 11.1
```


# Repo's Structure
```
|--cascade_fer: main directory for the source code of this project
|--tests: unit test files
|--environment.yml: yaml file to reproduce the conda environment used to build the project. To reproduce this conda environment, run `conda env create -f environment.yml`
|--pyproject.toml: config file for a poetry project, used to manage dependences and build the project using `poetry` command
|--requirements.txt: other package dependencies that are required to reproduce results. Once sourcing the conda environment, run `pip install -r requirements.txt`
```

# Installation
There are two methods: 1) install from the source, and 2) install from the wheel file.

## Install from Source
I use `poetry` to manage the whole project. For example, to build the project
```
poetry build
```
To install this package
```
poetry install
```
To update the project
```
poetry update
```
To add a new package dependenices
```
poetry add <package name>
```
For example
```
poetry add tensorflow-gpu==2.5
poetry add "opencv-python>=4.0.0"
```
This new package will be added to `pyproject.toml` for easy maintanence and upgrade. For example, to update dependences to their latest available versions, just simply run
```
poetry update
```

## Install from the Wheel File
```
git clone git@github.com:tynguyen/cascade_facial_emotion_recognition.git
pip install dist/cascade_fer-<version>.whl
```

## Usages
```
>>> from cascade_fer import CascadeFER
>>> cfer = CascadeFER()
>>> cfer(image)
```
To run a livestream demo, run
```
python tests/test_live_fer.py
```