# Coding Standards & Conventions

## **Code Quality Rules**
- **File Size**: Keep modules under 300 lines
- **CLI Interface**: Use Click for all scripts
- **Debug Output**: Use icecream (ic) instead of print
- **Error Handling**: Comprehensive try/catch with logging
- **Type Hints**: Use throughout for better code clarity

## **File Organization**
- **Utils**: Reusable functions in `src/utils/`
- **Configs**: Hydra YAML files in `configs/`
- **Scripts**: Entry points in `scripts/` with Fire CLI

## **Context Management**
- **Token Limit**: Keep under 50,000 tokens
- **Incremental Changes**: Provide only changed lines, not full files
- **Module Focus**: Implement one component at a time