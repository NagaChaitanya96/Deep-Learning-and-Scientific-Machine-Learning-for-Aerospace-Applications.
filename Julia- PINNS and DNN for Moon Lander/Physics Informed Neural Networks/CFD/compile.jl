using Weave
readdir()
jl_files = filter(endswith("jl"),readdir())

jmd_files =filter(endswith("jmd"),readdir())

pdf_files = filter(endswith("pdf"),readdir())

weave.(jmd_files)
# tangle.(jmd_files) will convert all the jmd files into code files.
