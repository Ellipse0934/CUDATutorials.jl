using Documenter, Literate
import CUDATutorials.make_dir

function generate_docs()
    outpath = normpath(joinpath(@__DIR__, "src"))
    make_dir(outpath)
    tutorial_path = normpath(joinpath(@__DIR__, "..", "src", "tutorials"))
    for (root, dirs, files) in walkdir(tutorial_path)
        for dir in dirs
            make_dir(joinpath(outpath, dir))
        end

        for file in files
            # TODO: non-regex solution ?
            m = match(Regex("^($tutorial_path)/(.+)"), root)

            if file[(end - 2):end] == ".jl"
                filename = file[begin:(end - 3)]
                Literate.markdown(joinpath(root, file), joinpath(outpath, String(m[2]));
                                    name = filename, execute=true, credit = false, documenter = true)
            else
                cp(joinpath(root, file), joinpath(outpath, String(m[2]), file);
                 force=true)
            end
        end
    end
end

@info "Building Tutorials"
generate_docs()

makedocs(
    sitename = "CUDATutorials.jl",
    authors = "Aditya Puranik",
    doctest = false,
    format = Documenter.HTML(prettyurls = false),
    pages = Any[
        "Home" => "index.md",
        "Introduction" => Any[
            "Introduction" => "introduction/01-Introduction.md",
            "Mandelbrot Set" => "./introduction/02-Mandelbrot_Set.md",
            "Shared Memory" => "./introduction/03-Shared_Memory.md",
            "Reduction" => "introduction/04-Reduction.md"
        ]
    ]
)