import numpy as np
import torch


def grokking_data_torch(p: int, op: str = '/', train_fraction: float = 0.5, device='cpu'):
    ## Validate inputs
    def safe_pow(base, exp, mod):
        return pow(int(base), int(exp), int(mod))

    # Define all operations with type-safe implementations
    operations = {
        '*': lambda a, b: (int(a) * int(b)) % p,
        '/': lambda a, b: (int(a) * safe_pow(b, p - 2, p)) % p,
        '+': lambda a, b: (int(a) + int(b)) % p,
        '-': lambda a, b: (int(a) - int(b)) % p,
        'x^3+xy^2+y': lambda a, b: (safe_pow(a, 3, p) + int(a) * safe_pow(b, 2, p) + int(b)) % p
    }

    if op not in operations:
        available_ops = list(operations.keys())
        raise ValueError(f"Unsupported operation '{op}'. Available: {available_ops}")

    # Generate all possible input pairs
    if op == '/':
        # For division, avoid b=0
        X = np.array([(a, b) for a in range(p) for b in range(1, p)])
    else:
        # For other operations, include all combinations
        X = np.array([(a, b) for a in range(p) for b in range(p)])

    # Compute corresponding outputs
    T = np.array([operations[op](a, b) for a, b in X])

    # Debug: Print some examples for verification
    if op == 'x^3+xy^2+y':
        print(f"Custom operation examples (mod {p}):")
        for i in range(min(5, len(X))):
            a, b = X[i]
            result = T[i]
           ## Verify the custom operation
            a_int, b_int = int(a), int(b)
            check = (safe_pow(a_int, 3, p) + a_int * safe_pow(b_int, 2, p) + b_int) % p
            print(f"  {a}^3 + {a}*{b}^2 + {b} ≡ {result} (mod {p}) [verify: {check}]")


    embed_tokens = {
        '*': p,
        '/': p + 1,
        '+': p + 2,
        '-': p + 3,
        'x^3+xy^2+y': p + 4,
        '=': p + 5
    }

    # Create input sequences: [a, op_token, b, equals_token]
    op_token = embed_tokens[op]
    equals_token = embed_tokens['=']
    X_seq = np.array([[a, op_token, b, equals_token] for (a, b) in X])

    # Split into train/test
    n_total = len(X_seq)
    n_train = int(train_fraction * n_total)

    # Random permutation for splitting
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train = X_seq[train_indices]
    T_train = T[train_indices]
    X_test = X_seq[test_indices]
    T_test = T[test_indices]

    # Convert to PyTorch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.long, device=device)
    T_train_torch = torch.tensor(T_train, dtype=torch.long, device=device)
    X_test_torch = torch.tensor(X_test, dtype=torch.long, device=device)
    T_test_torch = torch.tensor(T_test, dtype=torch.long, device=device)

    vocab_size = p + 6

    # print(f"Data generated for '{op}' mod {p}:")
    # print(f"  Total samples: {n_total}")
    # print(f"  Train: {len(X_train)} ({len(X_train) / n_total * 100:.1f}%)")
    # print(f"  Test: {len(X_test)} ({len(X_test) / n_total * 100:.1f}%)")
    # print(f"  Operation token: {op_token}")
    # print(f"  Equals token: {equals_token}")
    # print(f"  Vocabulary size: {vocab_size} (numbers 0-{p - 1}, op_tokens {p}-{p + 4}, equals_token={equals_token})")

    return X_train_torch, T_train_torch, X_test_torch, T_test_torch
