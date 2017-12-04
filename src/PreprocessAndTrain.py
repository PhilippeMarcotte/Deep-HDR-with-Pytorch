from PrepareTrainingData import distribute_training_data_preparation
from TrainersDeepHDR import DirectDeepHDRTrainer

distribute_training_data_preparation()
trainer = DirectDeepHDRTrainer()
trainer.train()
