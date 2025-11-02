using RAGTools
using Downloads
using JSON
using Serialization
using Transformers.HuggingFace
using Transformers.TextEncoders
using LinearAlgebra

const JOKE_DATASET_URL = "https://raw.githubusercontent.com/taivop/joke-dataset/master/wocka.json"
const MAX_JOKES = 1000   # Reduced to prevent memory issues
const BATCH_SIZE = 32    # Process embeddings in smaller batches

# Create a custom embedder using HuggingFace transformers
struct LocalTransformerEmbedder <: RAGTools.AbstractEmbedder
    encoder::Any
    model::Any
end

function RAGTools.get_embeddings(embedder::LocalTransformerEmbedder, texts::Vector{<:AbstractString}; verbose::Bool=false, cost_tracker=nothing, api_kwargs=nothing, kwargs...)
    # Log progress if verbose
    verbose && @info "Generating embeddings for $(length(texts)) texts"
    
    # Convert to concrete String type
    texts = collect(String, texts)
    num_texts = length(texts)
    
    # Process first batch to get embedding dimension
    first_batch = texts[1:min(BATCH_SIZE, num_texts)]
    encoded = encode(embedder.encoder, first_batch)
    output = embedder.model(encoded)
    embedding_dim = size(output.pooled, 1)
    
    # Initialize result matrix (embedding_dim × num_texts)
    all_embeddings = zeros(Float32, embedding_dim, num_texts)
    
    # Store first batch results
    first_batch_size = size(output.pooled, 2)
    for i in 1:first_batch_size
        vec = output.pooled[:, i]
        vec ./= norm(vec)
        all_embeddings[:, i] = vec
    end
    
    # Process remaining batches if any
    if num_texts > BATCH_SIZE
        for batch_start in (BATCH_SIZE + 1):BATCH_SIZE:num_texts
            batch_end = min(batch_start + BATCH_SIZE - 1, num_texts)
            batch = texts[batch_start:batch_end]
            
            verbose && @info "Processing batch $(batch_start:batch_end) of $num_texts"
            
            # Generate embeddings for batch
            encoded = encode(embedder.encoder, batch)
            output = embedder.model(encoded)
            embeddings = output.pooled
            
            # Process and store results
            for (i, col) in enumerate(1:size(embeddings, 2))
                vec = embeddings[:, col]
                vec ./= norm(vec)
                all_embeddings[:, batch_start + i - 1] = vec
            end
            
            # Force garbage collection to manage memory
            GC.gc()
        end
    end
    
    return all_embeddings
end

# Initialize the transformer model once
const TRANSFORMER_MODEL = let
    model = hgf"sentence-transformers/all-MiniLM-L6-v2"
    LocalTransformerEmbedder(model[1], model[2])
end

"""
Download and save jokes to text files
"""
function download_jokes(context_path::String)
    # Create directories
    docs_dir = joinpath(context_path, "jokes")
    mkpath(docs_dir)
    
    # Download dataset if not already present
    jokes_file = joinpath(context_path, "jokes.json")
    if !isfile(jokes_file)
        @info "Downloading joke dataset..."
        Downloads.download(JOKE_DATASET_URL, jokes_file)
    end
    
    # Load jokes
    jokes = open(jokes_file, "r") do f
        JSON.parse(f)
    end
    
    # Save each joke as a separate text file
    joke_files = String[]
    for (i, joke) in enumerate(jokes[1:min(MAX_JOKES, length(jokes))])
        joke_text = string(
            "Title: ", get(joke, "title", ""), "\n\n",
            get(joke, "body", ""), "\n\n",
            "Score: ", get(joke, "score", 0), "\n",
            "Upvotes: ", get(joke, "upvotes", 0)
        )
        
        # Save to file
        file_path = joinpath(docs_dir, "joke_$(i).txt")
        write(file_path, joke_text)
        push!(joke_files, file_path)
    end
    
    return joke_files
end

"""
Initialize the RAG engine with the given context path
"""
function init_rag(context_path::String)
    # Check for existing index
    index_path = joinpath(context_path, "index.jls")
    if isfile(index_path)
        @info "Loading existing index..."
        return deserialize(index_path)
    end
    
    # Prepare joke files
    joke_files = download_jokes(context_path)
    
    # Build the index using our local embedder
    @info "Building RAG index..."
    index = build_index(
        RAGTools.SimpleIndexer(),
        joke_files;
        embedder=TRANSFORMER_MODEL,
        chunker=RAGTools.TextChunker(),
        chunker_kwargs=(; sources=map(i -> "Joke$i", 1:length(joke_files)))
    )
    
    # Save index for future use
    @info "Saving index to $index_path"
    serialize(index_path, index)
    
    return index
end

"""
Retrieve relevant context for a query
"""
function retrieve_context(index, query::String)
    # Create a simple retriever with our configuration
    retriever = RAGTools.SimpleRetriever(;
        embedder=TRANSFORMER_MODEL,    # Use our local embedder for query embedding
        finder=RAGTools.CosineSimilarity()  # Use cosine similarity for finding matches
    )
    
    # Get relevant chunks using the configured retriever
    result = RAGTools.retrieve(
        retriever,
        index,
        query;
        top_k=5,          # Number of results to return
        top_n=10,         # Number of candidates to consider for reranking
        minimum_similarity=0.3,
        verbose=1
    )
    
    # Format the chunks into context
    context = ""
    if !isempty(result.context)
        # Get similarity scores from the candidates
        scores = if !isnothing(result.filtered_candidates)
            result.filtered_candidates.scores  # Use filtered candidates if available
        elseif !isnothing(result.emb_candidates)
            result.emb_candidates.scores      # Otherwise use embedding candidates
        else
            fill("N/A", length(result.context))
        end
        
        # Include source information and similarity scores
        for (i, (chunk, source)) in enumerate(zip(result.context, result.sources))
            score = scores isa Vector ? round(scores[i], digits=3) : "N/A"
            
            context *= """
            Joke $i (Score: $score):
            Source: $source
            $chunk
            
            """
        end
    end
    
    return context
end
