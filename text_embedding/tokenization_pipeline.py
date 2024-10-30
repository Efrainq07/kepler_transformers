import json
from transformers import AutoTokenizer
from tqdm import tqdm 

class TokenizationPipeline:
    def __init__(self, model_name: str, max_length: int = 512):
        # Initialize the tokenizer from the provided model name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length  # Set maximum token length


    def tokenize_from_dataloader(self, dataloader, output_file: str):
        """
        Tokenizes batches of entity descriptions from the DataLoader and saves the tokenized outputs to a file.
        """
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for batch in tqdm(dataloader, desc="Tokenizing", unit="batch"):
                tokenized_batch = self.tokenizer(batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.max_length)  # Apply token limit)
                for i in range(len(batch)):
                    token_dict = {
                        'input_ids': tokenized_batch['input_ids'][i].tolist(),
                        'attention_mask': tokenized_batch['attention_mask'][i].tolist()
                    }
                    # Write each tokenized entity description to the output file
                    json.dump(token_dict, outfile)
                    outfile.write('\n')

# Example usage:
# token_pipeline = TokenizationPipeline(model_name="bert-base-uncased")
# token_pipeline.tokenize_from_dataloader(dataloader=dataloader, output_file="tokenized_entities.json")
