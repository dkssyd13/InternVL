import sys
import json
import torch
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

sys.path.append("..")
from bert_score import score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
assert torch.cuda.is_available(), "CUDA is not available"
num_gpus = torch.cuda.device_count()
assert num_gpus >= 2, f"At least 2 GPUs required, but only {num_gpus} GPU(s) available"
logger.info(f'Found {num_gpus} GPU(s) available')

# Load Base Model
logger.info('Loading Base Model on GPU 0...')
BASE_MODEL_NAME = "OpenGVLab/InternVL3-1B"
base_model = AutoModel.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval().cuda(0)  # GPU 0

base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
logger.info('Base Model loaded successfully on GPU 0')

# Helper Functions
def build_transform(input_size=448):
    """Build transformation for video frames"""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_video(video_path, input_size=448):
    """Load video frames from directory using PIL"""
    try:
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise ValueError(f"{video_path=} does not exist")
        
        if not video_path.is_dir():
            raise ValueError(f"{video_path=} is not a directory")
        
        # Get all image files in the directory (sorted by name)
        frame_files = sorted([
            f for f in video_path.iterdir() 
        ])
        
        if len(frame_files) == 0:
            raise ValueError(f"{video_path=} has no image files")
        
        logger.debug(f"Found {len(frame_files)} frames in {video_path}")
        
        # Load and transform all frames using PIL
        transform = build_transform(input_size)
        pixel_values_list = []
        num_patches_list = []
        
        for frame_file in frame_files:
            img = Image.open(frame_file).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=1)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    
    except Exception as e:
        raise ValueError(f"Error loading video {video_path}: {e}")

def load_jsonl(path):
    """Load JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


if __name__ == "__main__":
    # === 1. Initialize paths ===
    MODELS_BASE_DIR = Path("../internvl_chat/work_dirs/internvl_chat_v3/internvl3_1b_dynamic_res_2nd_finetune_full")
    CHECKPOINTS = {
        "epoch_3": MODELS_BASE_DIR / "checkpoint-3297",
        "epoch_5": MODELS_BASE_DIR / "checkpoint-6595",
        "epoch_10": MODELS_BASE_DIR / "checkpoint-9892",
        "epoch_15": MODELS_BASE_DIR / "checkpoint-9892",  # 또는 정확한 checkpoint 번호
        "epoch_20": MODELS_BASE_DIR / "checkpoint-13180",
    }
    REFERENCE_DATA_PATH = Path("/storage/hdd1/internvl_data/playground/val_data")
    REFERENCE_LABEL_PATH = Path("../internvl_chat/annotation/hilab_vlm_val.jsonl")

    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        MODEL_PATH = checkpoint_path
        for path in [REFERENCE_DATA_PATH, REFERENCE_LABEL_PATH, MODEL_PATH]:
            if not path.exists():
                raise FileNotFoundError(f"{path=} does not exist")

        FINE_TUNED_PREDICTIONS = []
        BASE_PREDICTIONS = []
        REFERENCES = []
        SAMPLE_IDS = []
        VIDEO_NAMES = []

        RESULTS_PATH = Path(f"./results/bertscore_{checkpoint_name}.json")
        if RESULTS_PATH.exists():
            sys.exit(f"{RESULTS_PATH=} already exists")
        else:
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # ==============================


        # === 3. Load Model ===
        logger.info('Loading Fine-Tuned Model on GPU 1...')
        fine_tuned_model = AutoModel.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda(1)  # GPU 1

        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        logger.info('Fine-Tuned Model loaded successfully on GPU 1')
        # ==============================


        # === 4. Load Reference Data ===
        logger.info('Loading reference data...')
        reference_data = load_jsonl(REFERENCE_LABEL_PATH)
        logger.info(f'Loaded {len(reference_data)} samples')
        # ==============================


        # === 5. Generate Predictions ===
        logger.info('Generating predictions...')
        generation_config = dict(
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
        )

        for idx, item in enumerate(tqdm(reference_data, desc="Processing videos")):
            video_file = item['video']
            video_path = REFERENCE_DATA_PATH / video_file
            sample_id = item['id']
            
            if not video_path.exists():
                raise FileNotFoundError(f"{video_path=} does not exist")
            
            # Load video frames
            pixel_values, num_patches_list = load_video(video_path)
            
            if pixel_values is None:
                raise ValueError(f"{video_path=} failed to load. Pixel values are None")
            
            pixel_values = pixel_values.to(torch.bfloat16)
            
            # Process each conversation pair
            for conv in item['conversations']:
                if conv['from'] == 'human':
                    question: str = conv['value']
                    # Find corresponding answer
                    conv_idx = item['conversations'].index(conv)
                    if conv_idx + 1 < len(item['conversations']):
                        reference_answer = item['conversations'][conv_idx + 1]['value']
                    else:
                        continue
                    
                    # Prepare question with video frame tokens
                    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                    full_question = question.replace('<video>\n', video_prefix)
                    
                    # Generate predictions from both models
                    try:
                        with torch.no_grad():
                            # Base model prediction (GPU 0)
                            pixel_values_gpu0 = pixel_values.cuda(0)
                            base_response = base_model.chat(
                                base_tokenizer,
                                pixel_values_gpu0,
                                full_question,
                                generation_config,
                                num_patches_list=num_patches_list,
                                history=None,
                                return_history=False
                            )
                            # Fine-tuned model prediction (GPU 1)
                            pixel_values_gpu1 = pixel_values.cuda(1)
                            fine_tuned_response = fine_tuned_model.chat(
                                fine_tuned_tokenizer,
                                pixel_values_gpu1,
                                full_question,
                                generation_config,
                                num_patches_list=num_patches_list,
                                history=None,
                                return_history=False
                            )
                            
                        
                        FINE_TUNED_PREDICTIONS.append(fine_tuned_response)
                        BASE_PREDICTIONS.append(base_response)
                        REFERENCES.append(reference_answer)
                        SAMPLE_IDS.append(sample_id)
                        VIDEO_NAMES.append(video_file)
                        
                    except Exception as e:
                        raise ValueError(f"Error generating prediction for sample {idx}: {e}")

        logger.info(f'Generated {len(FINE_TUNED_PREDICTIONS)} predictions from each model')
        # ==============================


        # === 6. Calculate BERTScore ===
        logger.info('Calculating BERTScore...')
        if len(FINE_TUNED_PREDICTIONS) == 0:
            logger.error("No predictions generated!")
            sys.exit(1)

        # Calculate BERTScore for fine-tuned model
        logger.info('Calculating BERTScore for Fine-Tuned Model...')
        ft_P, ft_R, ft_F1 = score(  # type: ignore
            FINE_TUNED_PREDICTIONS,
            REFERENCES,
            lang='en',
            verbose=True,
            model_type='bert-base-uncased'
        )

        # Calculate average scores for fine-tuned model
        ft_avg_precision = ft_P.mean().item()  # type: ignore
        ft_avg_recall = ft_R.mean().item()  # type: ignore
        ft_avg_f1 = ft_F1.mean().item()  # type: ignore

        logger.info('Fine-Tuned Model BERTScore Results:')
        logger.info(f'  Precision: {ft_avg_precision:.4f}')
        logger.info(f'  Recall: {ft_avg_recall:.4f}')
        logger.info(f'  F1: {ft_avg_f1:.4f}')

        # Calculate BERTScore for base model
        logger.info('Calculating BERTScore for Base Model...')
        base_P, base_R, base_F1 = score(  # type: ignore
            BASE_PREDICTIONS,
            REFERENCES,
            lang='en',
            verbose=True,
            model_type='bert-base-uncased'
        )

        # Calculate average scores for base model
        base_avg_precision = base_P.mean().item()  # type: ignore
        base_avg_recall = base_R.mean().item()  # type: ignore
        base_avg_f1 = base_F1.mean().item()  # type: ignore

        logger.info('Base Model BERTScore Results:')
        logger.info(f'  Precision: {base_avg_precision:.4f}')
        logger.info(f'  Recall: {base_avg_recall:.4f}')
        logger.info(f'  F1: {base_avg_f1:.4f}')

        # Calculate improvement
        logger.info('\nImprovement (Fine-tuned vs Base):')
        logger.info(f'  Precision: {ft_avg_precision - base_avg_precision:+.4f}')
        logger.info(f'  Recall: {ft_avg_recall - base_avg_recall:+.4f}')
        logger.info(f'  F1: {ft_avg_f1 - base_avg_f1:+.4f}')
        # ==============================


        # === 7. Save Results ===
        logger.info('Saving results...')

        # Prepare predictions with id and video name
        fine_tuned_predictions_with_metadata = [
            {
                'id': SAMPLE_IDS[i],
                'video': VIDEO_NAMES[i],
                'prediction': FINE_TUNED_PREDICTIONS[i]
            }
            for i in range(len(FINE_TUNED_PREDICTIONS))
        ]

        base_predictions_with_metadata = [
            {
                'id': SAMPLE_IDS[i],
                'video': VIDEO_NAMES[i],
                'prediction': BASE_PREDICTIONS[i]
            }
            for i in range(len(BASE_PREDICTIONS))
        ]

        references_with_metadata = [
            {
                'id': SAMPLE_IDS[i],
                'video': VIDEO_NAMES[i],
                'reference': REFERENCES[i]
            }
            for i in range(len(REFERENCES))
        ]

        # Prepare per-sample scores with id only
        ft_per_sample_scores = [
            {
                'id': SAMPLE_IDS[i],
                'precision': ft_P[i].item(),  # type: ignore
                'recall': ft_R[i].item(),  # type: ignore
                'f1': ft_F1[i].item()  # type: ignore
            }
            for i in range(len(SAMPLE_IDS))
        ]

        base_per_sample_scores = [
            {
                'id': SAMPLE_IDS[i],
                'precision': base_P[i].item(),  # type: ignore
                'recall': base_R[i].item(),  # type: ignore
                'f1': base_F1[i].item()  # type: ignore
            }
            for i in range(len(SAMPLE_IDS))
        ]

        results = {
            'num_samples': len(FINE_TUNED_PREDICTIONS),
            'fine_tuned_model': {
                'scores': {
                    'precision': ft_avg_precision,
                    'recall': ft_avg_recall,
                    'f1': ft_avg_f1
                },
                'per_sample_scores': ft_per_sample_scores,
                'predictions': fine_tuned_predictions_with_metadata
            },
            'base_model': {
                'scores': {
                    'precision': base_avg_precision,
                    'recall': base_avg_recall,
                    'f1': base_avg_f1
                },
                'per_sample_scores': base_per_sample_scores,
                'predictions': base_predictions_with_metadata
            },
            'improvement': {
                'precision': ft_avg_precision - base_avg_precision,
                'recall': ft_avg_recall - base_avg_recall,
                'f1': ft_avg_f1 - base_avg_f1
            },
            'references': references_with_metadata
        }

        with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f'Results saved to {RESULTS_PATH}')
        logger.info('Done!')
        # ==============================