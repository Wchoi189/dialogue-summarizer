import torch
from icecream import ic
import sys
from pathlib import Path

# Add project root to path to allow imports
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from models.kobart_model import KoBARTSummarizationModel
from utils.config_utils import ConfigManager
from evaluation.metrics import RougeCalculator

def debug_generation(
    checkpoint_path: str,
    dialogue: str,
    ground_truth: str = None,
    generation_params: dict = None,
):
    """
    Loads a model checkpoint and tests generation on a single dialogue.
    """
    ic("--- Starting Generation Debug ---")
    ic(f"Checkpoint: {checkpoint_path}")

    # 1. Load Model and Config
    try:
        config_manager = ConfigManager()
        cfg = config_manager.load_config(config_name="kobart-base-v2.yaml")

        model = KoBARTSummarizationModel.load_from_checkpoint(checkpoint_path, cfg=cfg)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        tokenizer = model.tokenizer
        ic("Model and tokenizer loaded successfully.")
    except Exception as e:
        ic("Error loading model!", e)
        return

    # 2. Define Generation Parameters
    if generation_params is None:
        # Default parameters from your training config
        generation_params = cfg.training.generation
    ic(f"Using Generation Parameters: {generation_params}")

    # 3. Prepare Input and Generate
    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    prediction = ""
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_params,
        )
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 4. Display Results
    print("\n" + "="*50)
    print("📜 INPUT DIALOGUE:")
    print(dialogue)
    print("="*50)
    if ground_truth:
        print("🎯 GROUND TRUTH:")
        print(ground_truth)
        print(f"(Length: {len(ground_truth)} chars)")
        print("="*50)
    
    print("🤖 MODEL PREDICTION:")
    print(prediction)
    print(f"(Length: {len(prediction)} chars)")
    print("="*50)

    # 5. Evaluate
    if ground_truth:
        rouge_calc = RougeCalculator()
        scores = rouge_calc.calculate_rouge([prediction], [ground_truth])
        ic(scores)


if __name__ == "__main__":
    # --- Step 1: CONFIGURE YOUR TEST ---
    # Update this path to your best-performing model checkpoint
    MODEL_CHECKPOINT = "outputs/models/best-epoch=04-val_rouge_f=0.0000.ckpt"
    
    # Paste the problematic dialogue and its ground truth here
    TEST_DIALOGUE = "주제: 의사 상담 | 대화: #Person1# : 안녕하세요, 오늘 기분이 어떠세요? #Person2# : 요즘 숨쉬기가 힘들어요. #Person1# : 최근에 감기에 걸렸나요? #Person2# : 아니요, 특별히 아픈 곳은 없어요. #Person1# : 그럼 천식 검사를 위해 폐 전문의에게 가보는 게 좋겠어요. 알레르기가 있으신가요? #Person2# : 네, 몇 가지 있어요."
    GROUND_TRUTH_SUMMARY = "#Person2#는 숨쉬기 어려워합니다. 의사는 #Person2#에게 증상을 확인하고, 천식 검사를 위해 폐 전문의에게 가볼 것을 권합니다."
    
    # --- Step 2: EXPERIMENT WITH PARAMETERS ---
    # Modify these parameters to see how they affect the output
    custom_params = {
        "max_length": 256,
        "min_length": 15,
        "num_beams": 5,
        "repetition_penalty": 1.8,   # More aggressive penalty
        "length_penalty": 0.8,     # Penalize long outputs
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    # --- Step 3: RUN THE SCRIPT ---
    if not Path(MODEL_CHECKPOINT).exists():
        print(f"ERROR: Model checkpoint not found at '{MODEL_CHECKPOINT}'")
        print("Please update the 'MODEL_CHECKPOINT' variable in the script.")
    else:
        debug_generation(
            checkpoint_path=MODEL_CHECKPOINT,
            dialogue=TEST_DIALOGUE,
            ground_truth=GROUND_TRUTH_SUMMARY,
            generation_params=custom_params
        )