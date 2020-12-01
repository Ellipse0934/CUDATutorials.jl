module CUDATutorials

using CUDA, Literate

function make_dir(path)
    try 
        mkdir(path)
    catch e
        e.code == -17 && return true
        throw(e)
    end
end

"""
    generate_notebooks(outpath = pwd())
Output all tutorials as Jupyter Notebooks to `outpath`
"""
function generate_notebooks(outpath = pwd())
    outpath = joinpath(outpath, "notebook")
    make_dir(outpath)
    tutorial_path = joinpath(@__DIR__, "tutorials")
    for (root, dirs, files) in walkdir(tutorial_path)
        for dir in dirs
            make_dir(joinpath(outpath, dir))
        end

        for file in files
            # TODO: non-regex solution ?
            m = match(Regex("^($tutorial_path)/(.+)"), root)

            if file[(end - 2):end] == ".jl"
                Literate.notebook(joinpath(root, file), joinpath(outpath, String(m[2]));
                                    name = file, execute=false, credit = false)
            else
                cp(joinpath(root, file), joinpath(outpath, String(m[2]), file))
            end
        end
    end
end

end #module
