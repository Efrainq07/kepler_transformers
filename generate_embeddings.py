from text_embedding import create_description_dataloader, EmbeddingPipeline

description_dataloader = create_description_dataloader('.data/wikidata5m/wikidata5m_text.txt', 100)

embedding_model_name = "bert-base-uncased"

embedding_pipeline = EmbeddingPipeline(embedding_model_name)
embedding_pipeline.generate_embeddings(dataloader=description_dataloader, output_dir=".processed")

