"""
Inference predictor for dialogue summarization.
Handles model loading, prediction generation, and output processing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import pandas as pd
import torch
from icecream import ic
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from postprocessing.postprocessing import apply_post_processing
from data.dataset import InferenceDataset, create_collate_fn
from data.preprocessing import create_preprocessor
from models.base_model import BaseSummarizationModel

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandbRun

class DialoguePredictor:
    """Predictor for dialogue summarization inference."""
    
    def __init__(
        self,
        model: BaseSummarizationModel,
        cfg: DictConfig,
        device: Optional[str] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            cfg: Configuration
            device: Device to run inference on
        """
        self.model = model
        self.cfg = cfg
       
        
        # Setup device
        if device is None:
            device = cfg.get("device", "auto")
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessor
        self.preprocessor = create_preprocessor(cfg)
        
        # Setup generation config from inference config
        self.generation_config = self._setup_generation_config()
        
        ic(f"DialoguePredictor initialized on {self.device}")
    
    def _post_process_output(self, text: str) -> str:
        """Post-process generated text using the centralized cleaner."""
        post_cfg = self.cfg.get("postprocessing", {})
        
        # ✅ Call the single, centralized function
        return apply_post_processing(text, post_cfg)

    def _setup_generation_config(self) -> Dict[str, Any]:
        """Setup generation configuration from inference config."""
        
        # BEFORE (modular config, phase specific)
        # gen_cfg = self.inference_cfg.generation
        
        # AFTER (centralized config + experiment overrides)
        gen_cfg = self.cfg.generation
        
        config = {
            "max_length": gen_cfg.get("max_length", 100),
            "min_length": gen_cfg.get("min_length", 1),
            "num_beams": gen_cfg.get("num_beams", 4),
            "no_repeat_ngram_size": gen_cfg.get("no_repeat_ngram_size", 2),
            "early_stopping": gen_cfg.get("early_stopping", True),
            "do_sample": gen_cfg.get("do_sample", False),
            "length_penalty": gen_cfg.get("length_penalty", 1.0),
            "temperature": gen_cfg.get("temperature", 1.0),
            "top_k": gen_cfg.get("top_k", 50),
            "top_p": gen_cfg.get("top_p", 1.0),
            "repetition_penalty": gen_cfg.get("repetition_penalty", 1.0),
        }
        
        # Set special token IDs
        tokenizer = self.preprocessor.tokenizer
        config.update({
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        })
        
        return config
    
    def predict_single(self, dialogue: str) -> str:
        """
        Predict summary for a single dialogue.
        
        Args:
            dialogue: Input dialogue text
            
        Returns:
            Generated summary
        """
        # Preprocess input
        inputs = self.preprocessor.prepare_inputs(
            dialogue=dialogue,
            summary=None,
            is_inference=True
        )
        
        # Convert to tensors and move to device
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.generation_config
            )
        
        # Decode output
        generated_ids = outputs[0] if len(outputs.shape) > 1 else outputs
        
        # Get the setting from the config, defaulting to False for safety
        skip_tokens = self.cfg.postprocessing.get("skip_special_tokens", False)

        summary = self.preprocessor.decode_outputs(
            generated_ids.cpu().tolist(),
            skip_special_tokens=skip_tokens  # Use the variable from the config
        )
        
        # Post-process
        summary = self._post_process_output(summary)
        
        return summary
    
    def predict_batch(
        self,
        dialogues: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        Predict summaries for a batch of dialogues.
        
        Args:
            dialogues: List of dialogue texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated summaries
        """
        if batch_size is None:
             # BEFORE: (modular config, phase specific)
            # batch_size = self.inference_cfg.batch_size
            
            # AFTER (✅ Change this line):
            batch_size = self.cfg.dataset.eval_batch_size        
        # Ensure batch_size is not None
        if batch_size is None:
            batch_size = 1
        
        summaries = []
        
        # Process in batches
        progress_bar = tqdm(
            range(0, len(dialogues), batch_size),
            desc="Generating summaries",
            disable=not show_progress
        )
        
        for i in progress_bar:
            batch_dialogues = dialogues[i:i + batch_size]
            batch_summaries = self._predict_batch_internal(batch_dialogues)
            summaries.extend(batch_summaries)
        
        return summaries
    
    def _predict_batch_internal(self, dialogues: List[str]) -> List[str]:
        """Internal batch prediction method."""
        # Preprocess batch
        batch_inputs = self.preprocessor.batch_preprocess(
            dialogues=dialogues,
            summaries=None,
            is_inference=True
        )
        # BEFORE
        # # Convert to tensors
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     [torch.tensor(ids) for ids in batch_inputs.input_ids],
        #     batch_first=True,
        #     padding_value=self.preprocessor.tokenizer.pad_token_id or 0
        # ).to(self.device)
        
        # attention_mask = torch.nn.utils.rnn.pad_sequence(
        #     [torch.tensor(mask) for mask in batch_inputs.attention_mask],
        #     batch_first=True,
        #     padding_value=0
        # ).to(self.device)

        # Use torch.as_tensor() to make the type clear to the linter
        input_ids = torch.as_tensor(batch_inputs["input_ids"]).to(self.device)
        attention_mask = torch.as_tensor(batch_inputs["attention_mask"]).to(self.device)
        # Generate
        with torch.no_grad():
            outputs = self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.generation_config
            )
        # Get the setting from the config, defaulting to False for safety
        skip_tokens = self.cfg.postprocessing.get("skip_special_tokens", False)

        # Decode outputs
        summaries = []
        for output in outputs:
            summary = self.preprocessor.decode_outputs(
                output.cpu().tolist(),
                skip_special_tokens=skip_tokens
            )
            summary = self._post_process_output(summary)
            summaries.append(summary)
        
        return summaries
    
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict summaries for a DataFrame with dialogue column.
        
        Args:
            df: DataFrame with dialogue data
            
        Returns:
            DataFrame with predictions
        """
        ic(f"Predicting summaries for {len(df)} samples")
        
        # Get column names from config
        id_col = self.cfg.dataset.columns.id
        input_col = self.cfg.dataset.columns.input
        
        if id_col not in df.columns or input_col not in df.columns:
            raise ValueError(f"Required columns {id_col}, {input_col} not found in DataFrame")
        
        # Extract dialogues
        dialogues = df[input_col].tolist()
        
        # Generate predictions
        summaries = self.predict_batch(
            dialogues=dialogues,
            show_progress=True
        )
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            id_col: df[id_col],
            "summary": summaries
        })
        
        ic(f"Generated {len(summaries)} predictions")
        return result_df
    
    def predict_from_file(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path]
    ) -> None:
        """
        Predict summaries from input CSV file and save to output file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
        """
        # Load input data
        from utils.file_utils import FileManager
        file_manager = FileManager()
        
        df = file_manager.load_csv(input_file)
        ic(f"Loaded {len(df)} samples from {input_file}")
        
        # Generate predictions
        predictions_df = self.predict_dataframe(df)
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_manager.save_csv(predictions_df, output_path)
        ic(f"Saved predictions to {output_file}")
    
  
    
    def create_dataloader(self, df: pd.DataFrame) -> DataLoader:
        """
        Create DataLoader for inference.
        
        Args:
            df: DataFrame with test data
            
        Returns:
            DataLoader for inference
        """
        # Create inference dataset
        dataset = InferenceDataset(
            data=df,
            preprocessor=self.preprocessor,
            cfg=self.cfg.dataset
        )
        
        # Create collate function
        collate_fn = create_collate_fn(
            tokenizer=self.preprocessor.tokenizer,
            is_inference=True
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            
            # BEFORE (modular config, phase specific)
            # batch_size=self.inference_cfg.batch_size,
            # shuffle=False,
            # num_workers=self.inference_cfg.get("num_workers", 4),
            # pin_memory=self.inference_cfg.get("pin_memory", True),
            
            #  Point to the centralized dataset settings
            batch_size=self.cfg.dataset.eval_batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=self.cfg.dataset.pin_memory,
            collate_fn=collate_fn
        )
        
        return dataloader
    
    def predict_with_dataloader(self, dataloader: DataLoader) -> pd.DataFrame:
        """
        Generate predictions using DataLoader.
        
        Args:
            dataloader: DataLoader for inference
            
        Returns:
            DataFrame with predictions
        """
        all_predictions = []
        all_sample_ids = []
        
        # Get the setting from the config, defaulting to False for safety
        skip_tokens = self.cfg.postprocessing.get("skip_special_tokens", False)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sample_ids = batch["sample_ids"]
                
                # Generate
                outputs = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **self.generation_config
                )
                
                # Decode
                for i, output in enumerate(outputs):
                    summary = self.preprocessor.decode_outputs(
                        output.cpu().tolist(),
                        skip_special_tokens=skip_tokens
                    )
                    summary = self._post_process_output(summary)
                    
                    all_predictions.append(summary)
                    all_sample_ids.append(sample_ids[i])
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            self.cfg.dataset.columns.id: all_sample_ids,
            "summary": all_predictions
        })
        
        return result_df


     # ✅ ADD THE @classmethod DECORATOR AND CHANGE THE FIRST ARGUMENT TO `cls`
    @classmethod
    def create_predictor(
        cls,  # The first argument is now the class itself
        model_path: str,
        cfg: DictConfig,
        device: Optional[str] = None
    ) -> "DialoguePredictor": # Use quotes for forward reference to the class
        """
        Create predictor from model checkpoint.
        """
        from models.kobart_model import KoBARTSummarizationModel
        
        model = KoBARTSummarizationModel.load_from_checkpoint(
            model_path
        )
        
        # ✅ Use `cls` to create the new instance
        predictor = cls(model, cfg, device)
        
        return predictor 