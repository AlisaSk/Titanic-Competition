from model.trainer import ModelTrainer
from model.predictor import TitanicPredictor

def main():
    # 1. Train Model
    print("Training model...")
    trainer = ModelTrainer(train_path="data/train.csv", model_save_path="data/model.pkl")
    accuracy = trainer.train_with_cross_validation()
    print(f"Model trained with accuracy: {accuracy:.2f}")

    # 2. Predict test.csv
    print("Making predictions...")
    predictor = TitanicPredictor(model_path="data/model.pkl")
    predictions = predictor.run(test_path="data/test.csv", output_path="output/outsubmission.csv")
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()