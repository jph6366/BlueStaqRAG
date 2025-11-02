module LocalLLM

using RAGTools
using ArgParse

include("model.jl")
include("rag.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--query"
            help = "The query to process"
            required = true
        "--context-path"
            help = "Path to the context documents"
            default = "data/context"
        "--model-path"
            help = "Path to the Mistral model"
            required = true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    # Initialize model and RAG components
    model = init_model(args["model-path"])
    rag_engine = init_rag(args["context-path"])
    
    # Process query
    context = retrieve_context(rag_engine, args["query"])
    response = generate_response(model, context, args["query"])
    
    println(response)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
