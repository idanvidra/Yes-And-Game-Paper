# Yes-And-Game-Paper: Dataset for Improvised Story Co-Creation
This repository contains datasets associated with the paper:
"When Hallucinations are Good: Building AI Agents for Co-Creation of Improvised Stories"

## Overview
This dataset comprises textual 'Yes! And...' games, including both human-human (HH) and AI-human (AIH) interactions. These games were collected and analyzed as part of a study on computational creativity and human-AI collaboration in improvised storytelling.

## Repository Structure
Yes-And-Game-Paper/
│
└── Story Datasets/
    ├── aih_stories.json
    └── hh_stories.json

## Dataset Description
## AIH Stories (aih_stories.json)
This file contains the AI-Human interaction stories. Each story is represented as a JSON object with the following structure:
{
  "id": "AIH_X",
  "messages": [
    {"role": "human", "content": "..."},
    {"role": "ai", "content": "..."},
    ...
  ],
  "num_messages": N,
  "num_words": M
}

## HH Stories (hh_stories.json)
This file contains the Human-Human interaction stories. Each story is represented as a JSON object with the following structure:
{
  "id": "HH_X",
  "messages": [
    {"role": "human1", "content": "..."},
    {"role": "human2", "content": "..."},
    ...
  ],
  "num_messages": N,
  "num_words": M
}

## Usage
These datasets can be used for various research purposes, including but not limited to:

Analyzing patterns in human-human vs. human-AI collaborative storytelling
Training and evaluating AI models for improvised narrative generation
Studying turn-taking and coherence in collaborative storytelling

## Citation
If you use this dataset in your research, please cite our paper:
[Citation will be added upon publication]

## License
[Specify the license under which you're releasing this data, e.g., MIT, CC-BY, etc.]

## Contact
For any questions regarding this dataset or the associated paper, please contact:
[Your Name or Research Group]
[Your Institution]
[Contact Email]

##Acknowledgments
We thank all participants who contributed to the creation of these datasets, as well as [any funding bodies or institutions that supported this work].
