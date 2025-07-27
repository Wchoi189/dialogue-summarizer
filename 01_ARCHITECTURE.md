# Technical Architecture & Design Decisions

## **Core Technology Stack**
- **Framework**: PyTorch Lightning + Hydra
- **Model**: KoBART (`hyunwoongko/kobart`) for Korean dialogue summarization
- **CLI**: Click for all scripts

## **Critical Design Decisions**
1. **No Topic Dependency**: Model uses only `dialogue → summary`
2. **Exact Submission Format**: Must match `sample_submission.csv`
3. **Korean Text Support**: UTF-8 encoding throughout
4. **Data Split**: Use provided train/dev split as-is

## **Data Schema**
- **Train/Dev**: `fname,dialogue,summary,topic`
- **Test**: `fname,dialogue`
- **Submission**: `fname,summary` (with index)

## **File Paths**
- **Data**: `/home/wb2x/workspace/dialogue-summarizer/data`
- **Special Tokens**: `#Person1#`, `#Person2#`, etc.