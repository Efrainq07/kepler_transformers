from text_embedding import create_description_dataloader, EmbeddingPipeline

instruction = "Instruct: Given an article description, retrieve relevant articles with some relation to this article.\n\n Description:"
    

description_dataloader = create_description_dataloader('.data/wikidata5m/wikidata5m_text.txt', instruction, 100)

embedding_model_name = "dunzhang/stella_en_1.5B_v5"

embedding_pipeline = EmbeddingPipeline(embedding_model_name)
embedding_pipeline.generate_embeddings(dataloader=description_dataloader, output_dir=".processed")

