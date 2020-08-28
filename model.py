import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        # all layers except the last one (softmax)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        #print("features middle", features)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):

        super().__init__()
        self.n_hidden = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        # Embedding vector
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Define the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)

        # Define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):

        captions = captions[:, :-1]
        captions = self.embed(captions)

        # Concatenate the features and caption inputs
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        outputs, (hidden, cell) = self.lstm(inputs)

        # Convert LSTM outputs to word predictions
        outputs = self.fc(outputs)

        return outputs


# ##########################################################################################################

# Attention!
# states is a tuple consists of hidden and cell state
# states = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
#                   torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
                 
# ############################################################################################################
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
#         self.states = states
        output = []
        
        for word in range(max_len):
            # print("inputs", inputs.shape) # torch.Size([1, 1, 100])
            
            outputs, states = self.lstm(inputs, states)
            # print('lstm output shape ', outputs.shape) # torch.Size([1, 1, 256])
            # print('lstm states shape ', states[0].shape, states[1].shape)  # torch.Size([1, 1, 256])
            
            outputs = self.fc(outputs)
            # print('fc output shape ', outputs.shape) # torch.Size([1, 1, 9955])
            
            outputs = outputs.squeeze(1) 
            # we squeeze() to give the give the fc layer output into argmax()
            # print('output.squeeze(1) shape ', outputs.shape) # torch.Size([1, 9955])
            
            tensor_ids = outputs.argmax(dim=1) 
            # print('tensor_ids shape ', tensor_ids.shape) # torch.Size([1])
     
            output.append(tensor_ids.item()) # use item() to get the value of the tensor     
              
            if (tensor_ids == 1 or len(output) >= max_len):
                # if reached the max length or predicted the end token
                break
            else:
                inputs = self.embed(tensor_ids)
                # print("embed(tensor_ids)",  inputs.shape) # torch.Size([1, 100])

                inputs = inputs.unsqueeze(1)
                # print('next inputs shape ', inputs.shape) # torch.Size([1, 1, 100])
            
        return output

