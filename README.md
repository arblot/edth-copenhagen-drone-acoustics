# Drone Acoustics Hackathon

> The Helsing hackathon challenge for machine learning on drone acoustics.

## Challenge Prompt

Automated detection of threats is essential in facilitating early warning and situational awareness.
Acoustic detection complements other sensor modalities; while radar, optical, and infrared sensors can be
used for drone detection, each has limitations such as weather and obstructions.
Given the low infrastructure costs and ability for rapid deployment, acoustic sensing presents a suitable additional
layer of surveillance for modern defense strategies.

The problem is split into two phases.

Phase 1: 3-class prediction. We provide a small curated dataset of open-source acoustic recordings split into three
categories: background, drone, and helicopter. The challenge is to train a model to separate these three class from
their acoustic signatures.

# TODO(jearly): Document phase 2
Phase 2:

## Data

Sourced from: https://github.com/DroneDetectionThesis/Drone-detection-dataset (audio + video dataset)  
Paper: [A dataset for multi-sensor drone detection](https://www.sciencedirect.com/science/article/pii/S2352340921007976#!)

### Audio Dataset Details

While the GitHub provides both audio and video, we are only interested in the audio data.  
The challenge is to perform three-class classification (background/drone/helicopter) purely from audio.  
Audio is captured from a Boya BY-MM1 mini cardioid directional microphone.  
The provided audio in two channel L/R format, which has been automatically processed from a mono microphone.  
For each 2-channel 10-second file, we convert this into single channel (left or right) non-overlapping 5 second clips.  
This means each individual original file becomes four distinct files in our dataset.

From the paper:  
_The audio part has 90 ten-second files in wav-format with a sampling frequency of 44100 Hz.  
There are 30 files of each of the three output audio classes [background, drone, helicopter].  
The clips are annotated with the filenames themselves, e.g. DRONE_001.wav, HELICOPTER_001.wav, BACKGROUND_001.wav, etc.  
The audio in the dataset is taken from the videos or recorded separately.  
The background sound class contains general background sounds recorded outdoor in the acquisition location and
includes some clips of the sounds from the servos moving the pan/tilt platform where the sensors were mounted._
