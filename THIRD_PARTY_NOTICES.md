# Third-party notices

This source package uses the installed PyTorch, NumPy, pandas, and
scikit-learn libraries under their respective licenses. They are not vendored.

The frozen prediction model and its weights are supplied by the user. The
SE-LLM fusion adapter targets a model exposing `fuser.attention`, with null
conditioning defined by a zero alignment tensor before the existing
decomposition and fusion. No upstream model files or weights are redistributed.

Benchmark files must be obtained and used under the corresponding dataset
terms. No benchmark samples, scalers, forecasts, or result tables are included.
