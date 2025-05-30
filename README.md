# Yes-And-Game Dataset: Improvised Story Co-Creation

[![Paper](https://img.shields.io/badge/Paper-ICCC'25-blue)](https://computationalcreativity.net/iccc25/)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](#dataset-files)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This repository contains datasets from the paper **"Playing Along: Building AI Agents for Co-Creation of Improvised Stories"** accepted at the International Conference on Computational Creativity (ICCC) 2025.

## 📖 About

This dataset captures textual **"Yes! And..."** improvisational storytelling games between human pairs (HH) and AI agents with humans (AIH). The "Yes! And..." principle is a fundamental tenet of improvisation where participants must accept their partner's contributions and build upon them, creating collaborative narratives in real-time.

### Key Statistics
- **Total Stories**: 204 games
- **Human-Human (HH)**: 129 stories from 82 participants  
- **AI-Human (AIH)**: 75 stories from 45 participants
- **AI Model**: GPT-4 with specialized prompting for improvisational storytelling
- **Average Story Length**: ~13-18 sentences per game

## 🗂️ Repository Structure

```
Yes-And-Game-Paper/
├── README.md
├── LICENSE  
└── Story_Datasets/
    ├── aih_stories.json    # AI-Human collaborative stories
    └── hh_stories.json     # Human-Human collaborative stories
```

## 📊 Dataset Description

### Human-AI Stories (`aih_stories.json`)
Stories created through collaboration between human participants and GPT-4 agents.

**JSON Structure:**
```json
{
  "id": "AIH_001",
  "messages": [
    {
      "role": "human", 
      "content": "Remember when we went to that mysterious island?"
    },
    {
      "role": "ai", 
      "content": "Yes, and the trees there were singing ancient melodies!"
    },
    {
      "role": "human", 
      "content": "Yes! And we followed the music deeper into the forest."
    }
  ],
  "num_messages": 12,
  "num_words": 156
}
```

### Human-Human Stories (`hh_stories.json`)
Stories created through collaboration between pairs of human participants.

**JSON Structure:**
```json
{
  "id": "HH_001", 
  "messages": [
    {
      "role": "human1", 
      "content": "Do you remember that day at the beach?"
    },
    {
      "role": "human2", 
      "content": "Yes, and how the sand turned into quicksilver!"
    },
    {
      "role": "human1", 
      "content": "Yes, and we used it to build a time machine."
    }
  ],
  "num_messages": 14,
  "num_words": 142
}
```

## 🎯 Research Applications

This dataset enables research in:

- **Computational Creativity**: Studying AI's creative capabilities in improvisational contexts
- **Human-AI Collaboration**: Analyzing interaction patterns in creative partnerships  
- **Narrative Generation**: Training models for collaborative storytelling
- **Improvisation Studies**: Understanding real-time creative decision-making
- **Evaluation Frameworks**: Developing metrics for creative AI systems

## 📈 Evaluation Framework

Our study employed multiple evaluation approaches:

- **Human Ratings**: 453 evaluators assessed stories on creativity, interest, surprise, cohesiveness, and agreement
- **Computational Metrics**: Novelty (semantic distance), surprise (language model perplexity), and engagement measures
- **Turing Test**: Blind evaluation of story authorship (human vs. AI collaboration)

**Key Finding**: Stories co-created with AI were rated comparably to human-human collaborations and were indistinguishable in blind evaluation.

## 🔗 Related Materials

- **Paper**: "Playing Along: Building AI Agents for Co-Creation of Improvised Stories" (ICCC 2025)
- **Supplementary Information**: [Available here](https://tinyurl.com/yesandresearch)
- **Conference**: [ICCC 2025](https://computationalcreativity.net/iccc25/)

## 📄 Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{vidra2025playing,
  title={Playing Along: Building AI Agents for Co-Creation of Improvised Stories},
  author={Vidra, Idan Dov and Kimron, Gal and Noy, Lior and Shamir, Ariel},
  booktitle={Proceedings of the International Conference on Computational Creativity},
  year={2025}
}
```

## 📧 Contact

**Idan Dov Vidra**  
Reichman University  
📧 [idan.vidra@post.runi.ac.il](mailto:idan.vidra@post.runi.ac.il)

**Gal Kimron**  
Reichman University  
📧 [gal.kimron@post.runi.ac.il](mailto:gal.kimron@post.runi.ac.il)

**Ariel Shamir**  
Reichman University  
📧 [arik@runi.ac.il](mailto:arik@runi.ac.il)

**Lior Noy**  
Ono Academic College  
📧 [lior.noy@ono.ac.il](mailto:lior.noy@ono.ac.il)

## 🙏 Acknowledgments

We thank all participants who contributed to creating these datasets, the reviewers and organizers of ICCC 2025, and our institutions: Reichman University and Ono Academic College.

## 📋 License

This dataset is released under the [MIT License](LICENSE).

---

*For technical questions about the dataset structure or research methodology, please refer to our paper and supplementary materials.*
