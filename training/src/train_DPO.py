from swift.llm import rlhf_main
from swift.trainers import TrainerFactory

if __name__ == "__main__":
    TrainerFactory.TRAINER_MAPPING["dpo"] = "custom_trainer.SearchResultMaskDPOTrainer"
    rlhf_main()
