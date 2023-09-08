import torch
from whisper import Whisper
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable
import os
from loss import Loss
from metric import WER
from sklearn.model_selection import train_test_split
import math

class WhisperTrainer:
    def __init__(self,
                 token_size: int,
                 n_mel_channels: int,
                 n: int,
                 d_model: int,
                 heads: int,
                 d_ff: int,
                 activation: Callable[[torch.Tensor], torch.Tensor],
                 dropout_rate: float,
                 eps: float,
                 learning_rate: float = 3e-5,
                 device: str = 'cpu',
                 checkpoint: str = None) -> None:
        
        self.model = Whisper(
            token_size=token_size,
            n_mel_channels=n_mel_channels,
            n=n,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            activation=activation,
            dropout_rate=dropout_rate,
            eps=eps
        )

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)

        self.device = device
        self.checkpoint = checkpoint
        self.epoch = 0

        self.losses = []
        self.scores = []
        self.val_losses = []
        self.val_scores = []

        self.cost = Loss()
        self.metric = WER()

        if self.checkpoint is not None:
            self.load_model(self.checkpoint)

    def load_model(self, path: str, location: str = None):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=location)

            self.model.load_state_dict(checkpoint[ModelInfo.MODEL_STATE_DICT])
            self.optimizer.load_state_dict(checkpoint[ModelInfo.OPTIMIZER_STATE_DICT])
            self.epoch = checkpoint[ModelInfo.EPOCH]
            self.losses = checkpoint[ModelInfo.LOSS]
            self.scores = checkpoint[ModelInfo.METRIC]
            self.val_losses = checkpoint[ModelInfo.VAL_LOSS]
            self.val_scores = checkpoint[ModelInfo.VAL_METRIC]

            checkpoint = None

    def __save_model(self, path: str):
        torch.save({
            ModelInfo.MODEL_STATE_DICT: self.model.state_dict(),
            ModelInfo.OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
            ModelInfo.EPOCH: self.epoch,
            ModelInfo.LOSS: self.losses,
            ModelInfo.METRIC: self.scores,
            ModelInfo.VAL_LOSS: self.val_losses,
            ModelInfo.VAL_METRIC: self.val_scores
        }, path)
    
    def save_model(self, path: str):
        try:
            self.__save_model(path)
        except:
            self.__save_model("./model.pt")

    def build_dataset(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor, batch_size: int):
        return DataLoader(dataset=TensorDataset(encoder_inputs, decoder_inputs), batch_size=batch_size, shuffle=True)
    
    def fit(self, specs: torch.Tensor, texts: torch.Tensor, epochs: int = 1, batch_size: int = 1, mini_batch: int = 1, **validation):
        assert specs.size(0) == texts.size(0)

        self.model.train()

        use_validation = len(validation.keys()) != 0 and 'val_type' in validation

        if use_validation:
            if validation['val_type'] == 'kfold':
                num_folds = 2
                if 'num_folds' in validation and type(validation['num_folds']) == 'int':
                    num_folds = validation['num_folds']

                assert epochs >= num_folds

                samples_per_fold = math.ceil(specs.size(0)/num_folds)

                epochs = epochs // num_folds

                for fold in range(num_folds):
                    val_start_idx = fold * samples_per_fold
                    val_end_idx = (fold+1) * samples_per_fold
                    
                    spec_val_set, text_val_set = specs[val_start_idx: val_end_idx], texts[val_start_idx: val_end_idx]
                    spec_train_set = torch.cat((specs[:val_start_idx], specs[val_end_idx:]), dim=0)
                    text_train_set = torch.cat((texts[:val_start_idx], texts[val_end_idx:]), dim=0)

                    train_dataloader = self.build_dataset(spec_train_set, text_train_set, batch_size)
                    val_dataloader = self.build_dataset(spec_val_set, text_val_set, batch_size)

                    self.train(train_dataloader, epochs, mini_batch, val_dataloader)


            else:
                val_size = 0.1
                if 'val_size' in validation and type(validation['val_size']) == 'float':
                    val_size = validation['val_size']
                
                spec_train_set, spec_val_set, text_train_set, text_val_set = train_test_split(specs, texts, test_size=val_size, random_state=41)

                train_dataloader = self.build_dataset(spec_train_set, text_train_set, batch_size=batch_size)
                val_dataloader = self.build_dataset(spec_val_set, text_val_set, batch_size)

                self.train(train_dataloader, epochs, mini_batch, val_dataloader)

        else:
            dataloader = self.build_dataset(specs, texts, batch_size)
            self.train(dataloader, epochs, mini_batch)


        if self.checkpoint is not None:
            self.save_model(self.checkpoint)
        else:
            self.save_model('./model.pt')
    
    def step(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor, labels: torch.Tensor, training: bool = True):
        if training:
            self.optimizer.zero_grad()

        outputs = self.model(encoder_inputs, decoder_inputs)

        loss = self.cost(outputs, labels)
        if training:
            loss.backward()
            self.optimizer.step()

        _, preds = torch.max(outputs, dim=-1)
        score = self.metric.score(preds, labels)

        return loss.item(), score
    
    def train(self, dataloader: DataLoader, epochs: int, mini_batch: int, val_dataloader: DataLoader = None):
        total_batches = len(dataloader)

        batch_loss = 0.0
        batch_score = 0.0

        epoch_loss = 0.0
        epoch_score = 0.0

        for _ in range(epochs):
            count = 0

            for index, data in enumerate(dataloader):
                encoder_inputs = data[0].to(self.device)
                decoder_inputs = data[1][:, :-1].to(self.device)
                labels = data[1][:, 1:].to(self.device)

                loss, score = self.step(encoder_inputs, decoder_inputs, labels, training=True)
                count += 1

                batch_loss += loss
                batch_score += score

                if index%mini_batch == mini_batch-1 or index==total_batches-1:
                    print(f"Epoch {self.epoch+1} Batch: {index + 1} Loss: {(batch_loss/count):.4f} Score: {(batch_score/count):.4f}")

                    epoch_loss += batch_loss/count
                    epoch_score += batch_score/count

                    batch_loss = 0.0
                    batch_score = 0.0
                    count = 0.0
            print(f"Epoch: {self.epoch+1} Train Loss: {(epoch_loss/total_batches):.4f} Train Score: {(epoch_score/total_batches):.4f}")

            self.losses.append(epoch_loss/total_batches)
            self.scores.append(epoch_score/total_batches)

            epoch_loss = 0.0
            epoch_score = 0.0

            if val_dataloader is not None:
                self.validate(val_dataloader)
    
    def validate(self, dataloader: DataLoader, training: bool = True):
        total_batches = len(dataloader)
        total_loss = 0.0
        total_score = 0.0
        for _, data in enumerate(dataloader, 0):
            encoder_inputs = data[0].to(self.device)
            decoder_inputs = data[1][:, :-1].to(self.device)
            labels = data[1][:, 1:].to(self.device)

            loss, score = self.step(encoder_inputs, decoder_inputs, labels, training=False)

            total_loss += loss
            total_score += score

        if training:
            print(f"Epoch: {self.epoch+1} Validation Loss: {(total_loss/total_batches):.4f} Validation Score: {(total_score/total_batches):.4f}")
            self.val_losses.append(total_loss/total_batches)
            self.val_scores.append(total_score/total_batches)
        else:
            print(f"Evaluated Results: Validation Loss: {(total_loss/total_batches):.4f} Validation Score: {(total_score/total_batches):.4f}")
        
    def evaluate(self, specs: torch.Tensor, texts: torch.Tensor, batch_size: int = 1):
        self.model.eval()
        dataloader = self.build_dataset(specs, texts, batch_size)
        self.validate(dataloader, training=False)

    def predict(self, spec: torch.Tensor, max_length: int, start_token: int, end_token: int):
        self.model.eval()

        decoder_input = torch.tensor(start_token).unsqueeze(0)

        encoder_output = self.model.encoder(spec, None)

        for _ in range(max_length):
            decoder_output = self.model.decoder(decoder_input, encoder_output, None, None)

            _, pred = torch.max(decoder_output[:, -1, :], dim=-1)

            if pred == end_token:
                break

            decoder_input = torch.cat((decoder_input, pred.unsqueeze(0)), dim=-1)
        
        return decoder_input



class ModelInfo:
    MODEL_STATE_DICT = 'model_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
    EPOCH = 'epoch'
    LOSS = 'loss'
    METRIC = 'metric'
    VAL_LOSS = 'val_loss'
    VAL_METRIC = 'val_metric'