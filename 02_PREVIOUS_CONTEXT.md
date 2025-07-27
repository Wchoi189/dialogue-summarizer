# Updated Project Assessment - Data Structure Confirmation

Note: These are snippets of a long conversation.

## **Key Data Insights**

1. **Data Location**: Confirmed at `/home/wb2x/workspace/dialogue-summarizer/data`
2. **Korean Language**: Confirmed Korean dialogue data
3. **Data Structure**: 
   - **Train/Dev**: `fname,dialogue,summary,topic` (topic column added!)
   - **Test**: `fname,dialogue` 
   - **Submission**: `fname,summary`
4. **File Sizes**: 
   - Train: ~13.7MB (substantial dataset)
   - Dev: ~535KB 
   - Test: ~460KB


# Project Plan - Assessment

## **Key Design Decisions**

✅ **No Topic Dependency**: Model will only use `dialogue` → `summary`  
✅ **Use Dev Set As-Is**: Maintain the provided train/dev split  
✅ **Exact Submission Format**: Match the indexed CSV format precisely  
✅ **Topic for Analysis Only**: Use topic column for EDA and validation strategies  

## **Refined Project Structure**

```
dialogue_summarization/
├── configs/
│   ├── config.yaml                 # Main config with defaults
│   ├── model/
│   │   ├── kobart.yaml            # KoBART-specific config
│   │   └── solar_api.yaml         # Solar API config
│   ├── dataset/
│   │   └── dialogue_data.yaml     # Dataset configuration
│   ├── training/
│   │   ├── baseline.yaml          # Training hyperparameters
│   │   └── experiment.yaml        # Experimental settings
│   └── inference/
│       └── generation.yaml        # Generation parameters
├── src/
│   ├── data/
│   │   ├── dataset.py             # Dataset classes (dialogue→summary only)
│   │   ├── preprocessing.py       # Korean text preprocessing  
│   │   ├── datamodule.py          # Lightning DataModule
│   │   └── analysis.py            # Topic-based EDA utilities
│   ├── models/
│   │   ├── kobart_model.py        # KoBART Lightning module
│   │   ├── solar_api_model.py     # Solar API wrapper
│   │   └── base_model.py          # Abstract base model
│   ├── training/
│   │   ├── trainer.py             # Training orchestration
│   │   └── callbacks.py           # Custom callbacks
│   ├── evaluation/
│   │   ├── metrics.py             # ROUGE computation
│   │   └── evaluator.py           # Evaluation pipeline
│   ├── inference/
│   │   ├── predictor.py           # Inference pipeline
│   │   ├── generator.py           # Text generation
│   │   └── submission.py          # Submission file creation
│   └── utils/
│       ├── config_utils.py        # Hydra utilities
│       ├── wandb_utils.py         # WandB utilities  
│       ├── file_utils.py          # File I/O utilities
│       └── logging_utils.py       # Logging setup
├── scripts/
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Evaluation script
│   ├── inference.py               # Inference script
│   ├── create_submission.py       # Submission generation
│   └── eda.py                     # Exploratory data analysis
├── notebooks/
│   ├── topic_analysis.ipynb       # Topic distribution analysis
│   └── results_visualization.ipynb
├── environment.yml                
├── .gitignore
├── README.md
└── pyproject.toml                 
```

## **Core Implementation Strategy**

### **1. Data Pipeline**
- **Input**: `dialogue` column only (ignore topic at training time)
- **Output**: `summary` column  
- **Validation**: Use provided dev.csv as-is
- **Topic Usage**: Only for EDA and stratified analysis

### **2. Model Architecture**
- **Primary**: KoBART (dialogue → summary)
- **Comparison**: Solar API baseline
- **Focus**: Improve upon the existing 47.1244 ROUGE-F1 score

### **3. Submission Pipeline**
- **Critical**: Match exact format with index column
- **Output**: `fname,summary` with proper indexing
- **Validation**: Ensure submission format matches `sample_submission.csv`

## **Updated Configuration**

## **Implementation Phases**

### **Phase 2A: Core Infrastructure** 
1. Environment setup with Korean NLP support
2. Project structure creation  
3. Utilities (config, wandb, file handling)

### **Phase 2B: Data Pipeline**
1. Dataset classes (dialogue→summary only)
2. Korean text preprocessing 
3. Lightning DataModule with provided train/dev split

### **Phase 2C: Model Implementation** 
1. Base Lightning module
2. KoBART implementation matching baseline performance
3. Training pipeline with ROUGE monitoring

### **Phase 2D: Evaluation & Submission**
1. ROUGE evaluation pipeline
2. **Critical**: Submission file generation with exact format matching
3. Inference pipeline for test.csv

### **Phase 2E: Enhancements**
1. Solar API comparison
2. Topic-based analysis for insights
3. Performance optimizations

## **Ready to Proceed**

I'm now ready to implement **Phase 2A: Core Infrastructure**. 

**Shall I start with:**
1. `environment.yml` with Korean NLP dependencies
2. Project directory structure creation
3. Core utilities (`config_utils.py`, `wandb_utils.py`, etc.)

