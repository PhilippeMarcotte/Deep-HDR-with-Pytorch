from PrepareTrainingData import distribute_training_data_preparation
from DeepHDRTrainers import DirectDeepHDRTrainer

distribute_training_data_preparation()
trainer = DirectDeepHDRTrainer()
trainer.train()
