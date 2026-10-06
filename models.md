# Frozen backbone and endpoint adapter

FARR is a refinement model around a frozen endpoint forecaster. The public
release does not embed a SE-LLM checkpoint or a private model directory.

Implement a small local adapter with:

```python
class MyBackbone:
    def predict(self, history, condition):
        # condition is exactly "valid" or "null"
        # return a float array shaped [N, H, C]
        ...
```

Then run `scripts/generate_endpoints.py` once per split. The script calls the
same adapter twice on the same history batch and saves the matched arrays. The
adapter is responsible for loading the permitted checkpoint, applying the
dataset scaler, selecting the channel order, and implementing the actual null
condition described in the paper.

If the frozen predictor is trained locally rather than downloaded, keep its
training code and configuration in the project that owns that predictor. This
FARR repository only promises a complete refinement implementation and a
stable endpoint interface; it does not promise to reconstruct an unavailable
private checkpoint.
