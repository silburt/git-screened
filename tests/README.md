## git-screened Test Suite

### Running the Test Suite

Each file in the folder is a self-contained test script for the main functions
of `git-screened`.  To run them, use `pytest` on the command line, eg.

```
pytest test_features.py
```

### Sample Files Used By the Test Suite

- `sample_script.py`: Contains example code (taken from 
[silburt/DeepMoon](github.com/silburt/DeepMoon)) to test functions on.  
