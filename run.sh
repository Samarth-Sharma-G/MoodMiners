# create conda environment (must have installed conda) env name: audio-analysis
conda env create -f environment.yml
python -m ipykernel install --user --name audio-analysis --display-name "audio-analysis"