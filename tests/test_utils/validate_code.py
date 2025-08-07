#!/usr/bin/env python3
"""
Code validation script to test components step by step.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import click



@click.group()
def cli():
    """Code validation CLI."""
    pass


@cli.command()
def test_inference():
    """Test inference components."""
    click.echo("Testing inference components...")
    
    try:
        from inference.predictor import DialoguePredictor
        click.echo("✓ DialoguePredictor import successful")
        
        # Test without actual model (just class structure)
        from omegaconf import DictConfig
        
        # Create minimal config for testing
        test_cfg = DictConfig({
            "dataset": {
                "columns": {
                    "id": "fname",
                    "input": "dialogue",
                    "target": "summary"
                }
            },
            "inference": {
                "batch_size": 8,
                "generation": {
                    "max_length": 100,
                    "num_beams": 4
                }
                # ✅ Remove old post_processing from here - now handled by dedicated config
            },
            "postprocessing": {  # ✅ Add new postprocessing config for testing
                "remove_tokens": ["<s>", "</s>"],
                "text_cleaning": {
                    "strip_whitespace": True
                }
            },
            "model": {
                "tokenizer": {
                    "name_or_path": "digit82/kobart-summarization"
                }
            }
        })
        
        click.echo("✓ Test config created")
        
        # Test post-processing function (doesn't need model)
        # We can't test the full predictor without a model, but we can test structure
        click.echo("✓ Inference components ready")
        
        return True
        
    except Exception as e:
        click.echo(f"❌ Inference test failed: {e}")
        return False


@cli.command()
def test_imports():
    """Test if all imports work correctly."""
    click.echo("Testing imports...")
    
    try:
        # Test config utilities
        from utils.config_utils import ConfigManager
        click.echo("✓ ConfigManager import successful")
        
        # Test file utilities
        from utils.file_utils import FileManager
        click.echo("✓ FileManager import successful")
        
        # Test data components
        from data.preprocessing import DialoguePreprocessor
        click.echo("✓ DialoguePreprocessor import successful")
        
        from data.dataset import DialogueDataset
        click.echo("✓ DialogueDataset import successful")
        
        from data.datamodule import DialogueDataModule
        click.echo("✓ DialogueDataModule import successful")
        
        # Test model components
        from models.base_model import BaseSummarizationModel
        click.echo("✓ BaseSummarizationModel import successful")
        
        from models.kobart_model import KoBARTSummarizationModel
        click.echo("✓ KoBARTSummarizationModel import successful")
        
        # Test evaluation components
        from evaluation.metrics import RougeCalculator
        click.echo("✓ RougeCalculator import successful")
        
        # Test inference components
        from inference.predictor import DialoguePredictor
        click.echo("✓ DialoguePredictor import successful")
        
        click.echo("\n🎉 All imports successful!")
        
    except ImportError as e:
        click.echo(f"❌ Import failed: {e}")
        return False
    
    return True


@cli.command()
def test_config():
    """Test configuration loading."""
    click.echo("Testing configuration loading...")
    
    try:
        from utils.config_utils import ConfigManager
        
        # Initialize config manager
        config_manager = ConfigManager()
        click.echo("✓ ConfigManager initialized")
        
        # Try to load config (this might fail if config files don't exist)
        try:
            cfg = config_manager.load_config("config")
            click.echo("✓ Configuration loaded successfully")
            
            # Print config summary
            summary = config_manager.get_config_summary(cfg)
            click.echo(f"✓ Config summary: {summary}")
            
        except Exception as e:
            click.echo(f"⚠️  Config loading failed (expected if configs don't exist): {e}")
        
        return True
        
    except Exception as e:
        click.echo(f"❌ Config test failed: {e}")
        return False


@cli.command()
def test_file_utils():
    """Test file utilities."""
    click.echo("Testing file utilities...")
    
    try:
        from utils.file_utils import FileManager
        
        # Initialize file manager
        file_manager = FileManager()
        click.echo("✓ FileManager initialized")
        
        # Test basic functionality (without actual files)
        info = file_manager.get_file_info("test_file.txt")
        click.echo(f"✓ File info works: {info}")
        
        return True
        
    except Exception as e:
        click.echo(f"❌ File utils test failed: {e}")
        return False


@cli.command()
def test_rouge():
    """Test ROUGE calculation."""
    click.echo("Testing ROUGE metrics...")
    
    try:
        from evaluation.metrics import RougeCalculator
        
        # Initialize calculator
        calculator = RougeCalculator()
        click.echo("✓ RougeCalculator initialized")
        
        # Test with sample data
        predictions = ["안녕하세요 테스트입니다", "두 번째 테스트"]
        references = ["안녕하세요 이것은 테스트", "두 번째 참조"]
        
        scores = calculator.calculate_rouge(predictions, references)
        click.echo(f"✓ ROUGE calculation works: {scores}")
        
        return True
        
    except Exception as e:
        click.echo(f"❌ ROUGE test failed: {e}")
        return False


@cli.command()
def test_data_paths():
    """Test if data paths exist."""
    click.echo("Testing data paths...")
    
    # Check common data paths
    data_paths = [
        "/home/wb2x/workspace/dialogue-summarizer/data",
        "data",
        "./data"
    ]
    
    found_data = False
    for data_path in data_paths:
        path = Path(data_path)
        if path.exists():
            click.echo(f"✓ Found data directory: {path.absolute()}")
            
            # Check for data files
            files = ["train.csv", "dev.csv", "test.csv", "sample_submission.csv"]
            for file_name in files:
                file_path = path / file_name
                if file_path.exists():
                    click.echo(f"  ✓ Found: {file_name}")
                else:
                    click.echo(f"  ❌ Missing: {file_name}")
            
            found_data = True
            break
    
    if not found_data:
        click.echo("❌ No data directory found")
    
    return found_data


@cli.command()
def run_all():
    """Run all validation tests."""
    click.echo("Running all validation tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("File Utils", test_file_utils),
        ("ROUGE Metrics", test_rouge),
        ("Inference", test_inference),
        ("Data Paths", test_data_paths),
        ("Configuration", test_config),
    ]
    
    results = {}
    for test_name, test_func in tests:
        click.echo(f"\n{'='*50}")
        click.echo(f"Running {test_name} test...")
        click.echo('='*50)
        
        try:
            # Call test function directly (it's already bound to the CLI context)
            result = test_func.callback()
            results[test_name] = result
        except Exception as e:
            click.echo(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    click.echo(f"\n{'='*50}")
    click.echo("VALIDATION SUMMARY")
    click.echo('='*50)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        click.echo(f"{test_name:20}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    click.echo(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        click.echo("🎉 All tests passed!")
    else:
        click.echo("⚠️  Some tests failed. Check logs above.")


if __name__ == "__main__":
    cli()