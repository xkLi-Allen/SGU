import torch
import os
class abstract_model(torch.nn.Module):
    def __init__(self):
        super(abstract_model,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_model(self, save_path):
        # self.logger.info('saving models')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)



