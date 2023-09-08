import torch
from torch import Tensor

class WER:
    def __init__(self) -> None:
        pass

    def __init_score(self,len_ref: int, len_pred: int):
        scores = torch.zeros((len_ref+1, len_pred+1))
        for i in range(len_ref):
            scores[i+1][0] = i+1
        for j in range(len_pred):
            scores[0][j+1] = j+1

        return scores
    
    def calculate_score(self, ref: Tensor, pred: Tensor):
        ref = ref[ref!=0]
        pred = pred[pred!=0]

        len_ref = ref.size(0)
        len_pred = pred.size(0)

        score_matrix = self.__init_score(len_ref, len_pred)

        for i in range(1, len_ref+1):
            for j in range(1, len_pred+1):
                if ref[i-1] == pred[j-1]:
                    score_matrix[i][j] = score_matrix[i-1][j-1]
                else:
                    score_matrix[i][j] = min(score_matrix[i-1][j-1] + 1, score_matrix[i-1][j] + 1, score_matrix[i][j-1] + 1)
        return 1 - score_matrix[len_ref][len_pred]/len_ref

    def score(self, outputs: Tensor, labels: Tensor):
        batch_size = labels.size(0)
        accuracy = 0.0
        for batch in range(batch_size):
            accuracy += self.calculate_score(labels[batch], outputs[batch])

        accuracy = accuracy/batch_size

        return accuracy