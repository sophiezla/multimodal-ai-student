import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchinfo import summary

class CNN_Encoder(nn.Module): 
    def __init__(self, embed_size):
        super().__init__()

        # Use ResNet18 which is simpler and well-established
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        
        # Add a new FC layer to get the embedding size we want
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

    def forward(self, images):
        features = self.resnet(images)
        # Flatten the feature map/image
        features = self.flatten(features)
        # Pass through our new FC layer
        features = self.fc(features)
        # Apply dropout
        features = self.dropout(features)
        return features

class LSTM_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_features, captions, states=None):
        """
        Forward pass that can handle both training (with teacher forcing) and inference
        
        Args:
            encoder_features: Features from the CNN encoder
            captions: Caption tokens for teacher forcing
            states: (h, c) LSTM states (optional, for sequential inference)
            
        Returns:
            outputs: Word predictions at each time step
            states: LSTM hidden states (for inference continuation)
        """
        # Embed the captions
        embeddings = self.dropout(self.embed(captions))
        
        # Append the encoder features as the first "word" in the sequence
        embeddings = torch.cat((encoder_features.unsqueeze(0), embeddings), dim=0)
        
        # Pass through LSTM (returning states for sequential processing)
        lstm_out, states = self.lstm(embeddings, states)
        
        # Get predictions for each word
        outputs = self.fc(lstm_out)
        
        return outputs, states

class CNN_to_LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size):
        super().__init__()

        self.encoder = CNN_Encoder(embed_size)
        self.decoder = LSTM_Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        """
        Forward pass for training with teacher forcing
        """
        encoder_features = self.encoder(images)
        
        outputs, _ = self.decoder(encoder_features, captions)
        
        return outputs
    
def strength_test():
    # Create tensors that match the dimensions from your dataset
    image_tensor = torch.randn([32, 3, 224, 224])  # 32 images from batch
    captions = torch.randint(0, 10000, (20, 32))   # 20 words for 32 captions

    # Initialize model with smaller dimensions for quicker testing
    model = CNN_to_LSTM(embed_size=256, hidden_size=256, num_layers=1, vocab_size=10000)

    summary(model, input_data=[image_tensor, captions])

    # Forward pass
    output = model(image_tensor, captions)
    
    # Print output shape
    print("Output tensor shape:", output.shape)
    print("Test completed successfully!")

if __name__ == "__main__":
    strength_test()
    
