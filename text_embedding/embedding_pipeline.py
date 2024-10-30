import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

class EmbeddingPipeline:
    def __init__(self, model_name: str, device: torch.device, max_length: int = 512):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model = torch.nn.DataParallel(self.model)
        self.model.eval()  # Ensure the model is frozen
        self.max_length = max_length
        self.embeddings_list = []  # Store embeddings for batch processing
        self.checkpoint_count = 0

    def generate_embeddings(self, dataloader: DataLoader, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Iterate over DataLoader to generate embeddings
        for descriptions in tqdm(dataloader, total=len(dataloader), desc="Generating embeddings", unit="batch"):
            tokens = self.tokenizer(descriptions, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)

            with torch.no_grad():
                output = self.model(**tokens)
                embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()  # Take the mean as the embedding
            self.save_checkpoint(output_dir, embeddings)
            self.checkpoint_count+=1
        
        self.combine_checkpoints(output_dir)

    def save_checkpoint(self, output_dir, embeddings):
        """Save embeddings to a checkpoint file."""
        file_path = os.path.join(output_dir, f"{self.model_name}_embeddings_checkpoint_{self.checkpoint_count}.npz")
        np.savez_compressed(file_path, embeddings=embeddings)
        print(f"Checkpoint saved: {file_path}")
    
    def combine_checkpoints(self, output_dir):
        """Combine all checkpoint files into a single embeddings file."""
        all_embeddings = []
        for file in os.listdir(output_dir):
            if file.endswith('.npz'):
                file_path = os.path.join(output_dir, file)
                checkpoint_data = np.load(file_path)
                all_embeddings.append(checkpoint_data['embeddings'])

        # Concatenate all embeddings and save to a single file
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        combined_file_path = os.path.join(output_dir, f"{self.model_name}_combined_embeddings.npz")
        np.savez_compressed(combined_file_path, embeddings=final_embeddings)
        print(f"Combined embeddings saved: {combined_file_path}")

# Example usage:
# dataset = EntityDescriptionDataset(file_path="path/to/your/file.txt")
# dataloader = DataLoader(dataset, batch_size=32, num_workers=4)  # Adjust batch size and number of workers
# pipeline = EmbeddingPipeline(model_name="bert-base-uncased", embedding_dim=768, device=torch.device("cuda"))
# pipeline.generate_embeddings(dataloader=dataloader, output_dir="entity_embeddings/")
