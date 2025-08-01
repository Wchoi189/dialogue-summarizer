"""
KoBART model implementation for Korean dialogue summarization.
Based on BART architecture with Korean language support.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from icecream import ic
from omegaconf import DictConfig
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration

from .base_model import BaseSummarizationModel

logger = logging.getLogger(__name__)


class KoBARTSummarizationModel(BaseSummarizationModel):
    """KoBART model for dialogue summarization."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize KoBART model.
        
        Args:
            cfg: Complete configuration
        """
        super().__init__(cfg)
        
        # Setup tokenizer first (needed for model initialization)
        self._setup_tokenizer()
        
        # Setup model
        self._setup_model()
        
        # Add assertions to ensure components are initialized
        assert self.model is not None, "Model not initialized"
        assert self.tokenizer is not None, "Tokenizer not initialized"
        
        # Resize token embeddings if special tokens were added
        if hasattr(self.tokenizer, 'vocab_size'):
            original_vocab_size = self.model.config.vocab_size
            current_vocab_size = len(self.tokenizer)
            
            if current_vocab_size > original_vocab_size:
                ic(f"Resizing token embeddings: {original_vocab_size} -> {current_vocab_size}")
                self.model.resize_token_embeddings(current_vocab_size)
        
        ic(f"KoBARTSummarizationModel initialized with {self.get_parameter_count()} parameters")
    
    def _setup_tokenizer(self) -> None:
        """Setup KoBART tokenizer with special tokens."""
        tokenizer_cfg = self.model_cfg.tokenizer
        model_name = tokenizer_cfg.name_or_path
        
        ic(f"Loading tokenizer: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=tokenizer_cfg.get("use_fast", True)
        )
        
        # # ✅ CHANGE: Get special tokens from dataset config instead of model config
        # dataset_cfg = self.cfg.dataset
        # additional_tokens = dataset_cfg.preprocessing.get("special_tokens", [])
        
        # if additional_tokens:
        #     # Ensure tokens are strings
        #     additional_tokens = [str(token) for token in additional_tokens]
            
        #     # ✅ CRITICAL FIX: Add these as regular vocabulary tokens, not special tokens.
        #     # This prevents `decode(skip_special_tokens=True)` from removing them.
        #     num_added = self.tokenizer.add_tokens(additional_tokens)
        #     # ic(f"Added {num_added} new tokens to vocabulary: {additional_tokens}")

        # Verify special tokens
        special_tokens = {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
        }
        # ic(f"Special tokens: {special_tokens}")
    
    def _setup_model(self) -> None:
        """Setup KoBART model."""
        model_name = self.model_cfg.model_name_or_path
        
        ic(f"Loading model: {model_name}")
        
        # Load model configuration
        config = BartConfig.from_pretrained(model_name)

        # Update config with custom parameters if specified  
        if "parameters" in self.model_cfg:
            params = self.model_cfg.parameters
            
            # Update relevant config parameters
            config_updates = {
                "encoder_layers": params.get("encoder_layers"),
                "decoder_layers": params.get("decoder_layers"),
                "encoder_attention_heads": params.get("encoder_attention_heads"),
                "decoder_attention_heads": params.get("decoder_attention_heads"),
                "encoder_ffn_dim": params.get("encoder_ffn_dim"),
                "decoder_ffn_dim": params.get("decoder_ffn_dim"),
                "d_model": params.get("d_model"),
            }
            
            for key, value in config_updates.items():
                if value is not None:
                    setattr(config, key, value)
                    ic(f"Updated config.{key} = {value}")
        
        # Load the model
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name,
            config=config
        )
        
        # Configure model for training/inference
        training_mode = self.model_cfg.get("training_mode", {})
        
        # Gradient checkpointing
        if training_mode.get("gradient_checkpointing", False) and self.model is not None:
            self.model.gradient_checkpointing_enable()
            ic("Gradient checkpointing enabled")
        
        # Cache usage
        if self.model is not None:
            self.model.config.use_cache = training_mode.get("use_cache", False)
        
        # Model compilation (PyTorch 2.0+)
        compile_cfg = self.model_cfg.get("compile", {})
        if compile_cfg.get("enabled", False):
            try:
                ic("Attempting torch.compile...")
                
                # Test different compile modes
                mode = compile_cfg.get("mode", "default")
                dynamic = compile_cfg.get("dynamic", False)
                
                ic(f"Compile settings: mode={mode}, dynamic={dynamic}")
                
                self.model = torch.compile(
                    self.model,
                    mode=mode,
                    dynamic=dynamic,
                    fullgraph=compile_cfg.get("fullgraph", False)
                )
                ic("✅ torch.compile successful!")
                
            except Exception as e:
                ic(f"❌ torch.compile failed: {e}")
                ic("Continuing without compilation...")
        else:
            ic("torch.compile disabled in config")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        """
        Forward pass through KoBART model.
        
        Args:
            input_ids: Encoder input token IDs [batch_size, seq_len]
            attention_mask: Encoder attention mask [batch_size, seq_len]
            decoder_input_ids: Decoder input token IDs [batch_size, target_len]
            decoder_attention_mask: Decoder attention mask [batch_size, target_len]
            labels: Target labels for loss computation [batch_size, target_len]
            **kwargs: Additional arguments
            
        Returns:
            Model outputs including loss and logits
        """
        assert self.model is not None
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            use_cache=self.model_cfg.get("inference_mode", {}).get("use_cache", True),
            output_attentions=self.model_cfg.get("inference_mode", {}).get("output_attentions", False),
            output_hidden_states=self.model_cfg.get("inference_mode", {}).get("output_hidden_states", False),
            **kwargs
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate summaries using KoBART.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        # Set model to inference mode
        inference_mode = self.model_cfg.get("inference_mode", {})
        assert self.model is not None, "Model not initialized"
        self.model.config.use_cache = inference_mode.get("use_cache", True)
        
        # Merge generation config with kwargs
        gen_kwargs = {**self.generation_config, **kwargs}
        
        # Set special token IDs
        gen_kwargs.update({
            "pad_token_id": self.tokenizer.pad_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        })
        
        assert self.model is not None, "Model not initialized"
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        return outputs
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter count."""
        if self.model is None:
            return {"total": 0, "trainable": 0}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get component-wise breakdown
        encoder_params = sum(p.numel() for p in self.model.model.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.model.model.decoder.parameters())
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "encoder": encoder_params,
            "decoder": decoder_params,
            "embedding": sum(p.numel() for p in self.model.model.shared.parameters())
        }
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for fine-tuning."""
        assert self.model is not None, "Model not initialized"
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        
        ic("Encoder parameters frozen")
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        assert self.model is not None, "Model not initialized"
        for param in self.model.model.encoder.parameters():
            param.requires_grad = True
        
        ic("Encoder parameters unfrozen")
    def freeze_decoder(self) -> None:
        """Freeze decoder parameters for fine-tuning."""
        assert self.model is not None, "Model not initialized"
        for param in self.model.model.decoder.parameters():
            param.requires_grad = False
        
        ic("Decoder parameters frozen")
    def unfreeze_decoder(self) -> None:
        """Unfreeze decoder parameters."""
        assert self.model is not None, "Model not initialized"
        for param in self.model.model.decoder.parameters():
            param.requires_grad = True
        
        ic("Decoder parameters unfrozen")
    def get_encoder_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get encoder embeddings for analysis.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Encoder hidden states
        """
        assert self.model is not None, "Model not initialized"
        with torch.no_grad():
            encoder_outputs = self.model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        return encoder_outputs.last_hidden_state
    
    def compute_perplexity(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute perplexity for generated text.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            
        Returns:
            Perplexity score
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            perplexity = torch.exp(loss).item()
            return perplexity
            
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model and tokenizer.
        
        Args:
            save_directory: Directory to save model
        """
        ic(f"Saving model to: {save_directory}")
        
        assert self.model is not None, "Model not initialized"
        assert self.tokenizer is not None, "Tokenizer not initialized"
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        ic("Model and tokenizer saved successfully")
        
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        ic("Model and tokenizer saved successfully")
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Any,
        map_location: Any = None,
        hparams_file: Optional[Union[str, Path]] = None,
        strict: Optional[bool] = True,
        **kwargs,
    ) -> "KoBARTSummarizationModel":
        """
        Load model from a PyTorch Lightning checkpoint, with an option to override
        the configuration for inference.
        """
        ic(f"Loading model from checkpoint: {checkpoint_path}")

        # Extract the complete 'cfg' object if it was passed. This is the new config
        # we want to use for the current task (e.g., inference).
        new_cfg = kwargs.pop('cfg', None)

        # Let the parent method load the model. It will use the hyperparameters
        # saved within the checkpoint file to initialize the model.
        # Any remaining `kwargs` will be used to override individual hparams.
        model = super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs
        )
        ic("Model instance created from checkpoint using its saved hyperparameters.")

        # If a new config (`new_cfg`) was provided, we manually update the
        # configuration of the loaded model. This is crucial for inference,
        # where we might use different settings (e.g., for generation).
        if new_cfg:
            ic("A new configuration was provided. Overriding the model's loaded config.")
            model.cfg = new_cfg
            model.model_cfg = new_cfg.model
            model.training_cfg = new_cfg.training
            
            # Re-initialize parts of the model that depend on the new configuration.
            # For inference, the generation config is especially important.
            model.generation_config = model._setup_generation_config()
            ic("Model configuration updated for the current task.")

        ic("Model loaded from checkpoint successfully and is ready.")
        return model


def create_kobart_model(cfg: DictConfig) -> KoBARTSummarizationModel:
    """
    Create KoBART model instance.
    
    Args:
        cfg: Complete configuration
        
    Returns:
        KoBART model instance
    """
    return KoBARTSummarizationModel(cfg)


def load_pretrained_kobart(
    model_path: str,
    cfg: DictConfig,
    device: Optional[str] = None
) -> KoBARTSummarizationModel:
    """
    Load pretrained KoBART model.
    
    Args:
        model_path: Path to pretrained model
        cfg: Configuration
        device: Device to load model on
        
    Returns:
        Loaded model instance
    """
    # Update config with model path
    cfg = cfg.copy()
    cfg.model.model_name_or_path = model_path
    cfg.model.tokenizer.name_or_path = model_path
    
    # Create model
    model = create_kobart_model(cfg)
    
    # Move to device if specified
    if device:
        model = model.to(device)
    
    ic(f"Pretrained KoBART loaded from: {model_path}")
    return model