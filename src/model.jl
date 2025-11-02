using LlamaCpp

"""
Model configuration struct to hold inference parameters
"""
struct ModelConfig
    ctx_size::Int
    temp::Float32
    top_p::Float32
    max_tokens::Int
    n_threads::Int
end

"""
Default configuration for the model
"""
const DEFAULT_CONFIG = ModelConfig(
    2048,                # context size
    0.7,                # temperature
    0.9,                # top_p
    256,                # max tokens to generate (reduced for faster responses)
    4                   # use 4 threads for better stability
)

"""
Initialize the Mistral model from the given path
"""
function init_model(model_path::String, config::ModelConfig=DEFAULT_CONFIG)
    if !isfile(model_path)
        throw(ArgumentError("Model file not found at: $model_path"))
    end
    
    # Return the model path and config for use in generate_response
    return (model_path=model_path, config=config)
end

"""
Format the prompt with context and query
"""
function format_prompt(context::String, query::String)
    # Format following Mistral's typical instruction format
    return """
    [INST]
    Use the following context to answer the query:
    
    Context:
    $context
    
    Query:
    $query
    [/INST]
    """
end

"""
Custom Llama answerer for RAG using LlamaCpp.jl
"""
struct LlamaAnswerer <: RAGTools.AbstractAnswerer
    model_path::String
    config::ModelConfig
end

"""
Generate an answer using the Llama model
"""
function RAGTools.answer!(answerer::LlamaAnswerer, index::RAGTools.AbstractDocumentIndex, result::RAGTools.RAGResult; verbose::Bool=false, kwargs...)
    # Join context into a single string
    context_str = join(result.context, "\n\n")
    
    # Format the prompt with context and query
    prompt = """
    [INST]
    Use the following context to answer the query:
    
    $context_str
    
    Query:
    $(result.question)
    [/INST]
    """
    
    # Build llama.cpp arguments - removed stop tokens since they're not supported
    args = `--ctx-size $(answerer.config.ctx_size) \
            --threads $(answerer.config.n_threads) \
            --temp $(answerer.config.temp) \
            --top-p $(answerer.config.top_p) \
            -n $(answerer.config.max_tokens) \
            --repeat-penalty 1.1`
    
    verbose && @info "Generating answer with LlamaCpp..."
    
    # Generate response using run_llama
    response = run_llama(;
        model=answerer.model_path,
        prompt=prompt,
        nthreads=answerer.config.n_threads,
        ctx_size=answerer.config.ctx_size,
        args=args
    )
    
    # Clean up response
    answer = strip(response)
    answer = replace(answer, r"\[/?INST\]" => "")  # Remove [INST] tags
    answer = replace(answer, "</s>" => "")        # Remove end token
    
    # Store both raw and final answer
    result.answer = response
    result.final_answer = strip(answer)
    
    return result
end

"""
Generate a response given the context and query
"""
function generate_response(model_tuple, context::String, query::String)
    # Create a simple index from our chunks
    chunks = split(context, "\n\n")
    sources = ["context_$i" for i in 1:length(chunks)]
    index = RAGTools.ChunkEmbeddingsIndex(
        :local_index,
        chunks,
        nothing,  # no embeddings needed for direct generation
        nothing,  # no tags
        nothing,  # no tags vocab
        sources,
        nothing   # no extras
    )
    
    # Create RAG result with query
    result = RAGTools.RAGResult(;
        question=query,
        context=chunks,
        sources=sources
    )
    
    # Set up generator with our custom answerer
    generator = RAGTools.SimpleGenerator(;
        answerer=LlamaAnswerer(model_tuple.model_path, model_tuple.config)
    )
    
    # Generate response using RAGTools' pipeline
    RAGTools.generate!(generator, index, result;
        verbose=1
    )
    
    return result.final_answer
end
