# ASL 2024 - tSNE

## Changelog

| What          | Speedup(n=500)    | Comment   |
|---------------|-------------------|-----------|
| Baseline      | 1.0               | Baseline based on reference script. |
| Online Q      | 3.84              | Inline the computation of the q matrix instead of first calculating it. |
| Compress P    | ?                 | ? |

## Getting Started

### Compile
```bash
./compile.sh
```

### Test
```bash
./test.sh
```

### Measure
```bash
./measure.sh
```

## Project Structure
* src: The source code of our high-performance implementation
* data: Some static data for testing and measuring
* tools: Some python tooling to generate seeds, data, plots, ... Also contains the reference python implementation
