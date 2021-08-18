import torch
import timm

class AttributeModel():
    def __init__(self, backbone : str, no_attributes : int, device = torch.device('cpu'), pretrain : bool = True) -> None:
        self.model = timm.create_model(backbone, pretrained=pretrain, num_classes=no_attributes)
        self.model.to(device)

    def save_checkpoint(self, save_name='modelCheckpoint.pt'):
        torch.save(self.model.state_dict(), save_name)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)

    def __call__(self, image_batch):
        """Used during training"""
        logits = self.model(image_batch)
        preds = torch.sigmoid(logits)
        return preds

    def match_attributes(self, image1, image2):
        """produce similarity score of image attributes of input pair of images"""
        attributes1 = self.model(image1)
        attributes2 = self.model(image2)
        score = self.compute_similarity(attributes1, attributes2)
        return score
    
    def compute_similarity(self):
        pass