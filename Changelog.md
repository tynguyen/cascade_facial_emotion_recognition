# Changelog
All major changes are listed here
# [0.1.2] Nov 7th 2021
# Added
- [] A stand-alone fer that takes a raw input image and outputs the dominant emotion over a duration

# Changed
- [] Simplify live facial emotion detection using the stand-alone fer

# [0.1.1] Oct 24th 2021
## Added
- [x] Live facial emotion detection
- [x] Use accumulative frames to determine the major emotion
- [x] Group emotions into positive and negative

## Changed
- [x] Remove local shape_predictor_file. Download if necessary

# [0.1.0] Oct 24th 2021
## Added
- [x] CV-based face and facial landmark detection with livestream input
Run:
```
poetry run tests/test_find_facial_landmarks.py
```