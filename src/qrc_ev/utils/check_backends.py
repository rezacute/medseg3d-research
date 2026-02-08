"""Backend verification script for quantum computing frameworks.

This script verifies that quantum backends are properly installed and can
execute simple circuits. It's useful for debugging installation issues and
confirming that the environment is correctly configured.
"""

import sys
from typing import Any, Dict, List, Tuple

import numpy as np


def check_pennylane() -> Tuple[bool, str]:
    """Verify PennyLane installation and basic functionality.

    Returns:
        Tuple of (success: bool, message: str) indicating whether PennyLane
        is working correctly and a descriptive message.
    """
    try:
        import pennylane as qml

        version = qml.__version__
        
        # Test basic circuit execution
        dev = qml.device("default.qubit", wires=2)
        
        @qml.qnode(dev)
        def simple_circuit() -> tuple:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
        
        result = simple_circuit()
        
        # Verify result is reasonable (should be close to [0, 0] for Bell state)
        if not isinstance(result, (tuple, list, np.ndarray)):
            return False, f"PennyLane v{version}: Unexpected result type {type(result)}"
        
        if len(result) != 2:
            return False, f"PennyLane v{version}: Expected 2 measurements, got {len(result)}"
        
        # Check that values are in valid range [-1, 1]
        if not all(-1.0 <= val <= 1.0 for val in result):
            return False, f"PennyLane v{version}: Measurement values out of range: {result}"
        
        return True, f"PennyLane v{version}: ✓ Installed and functional"
    
    except ImportError:
        return False, "PennyLane: ✗ Not installed (pip install pennylane)"
    except Exception as e:
        return False, f"PennyLane: ✗ Error during execution: {str(e)}"


def check_lightning() -> Tuple[bool, str]:
    """Verify PennyLane Lightning plugin installation.

    Returns:
        Tuple of (success: bool, message: str) indicating whether Lightning
        is available and a descriptive message.
    """
    try:
        import pennylane as qml
        
        # Try to create a lightning.qubit device
        dev = qml.device("lightning.qubit", wires=2)
        
        @qml.qnode(dev)
        def simple_circuit() -> Any:
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        
        result = simple_circuit()
        
        if not isinstance(result, (float, np.floating)):
            return False, f"Lightning: Unexpected result type {type(result)}"
        
        if not -1.0 <= result <= 1.0:
            return False, f"Lightning: Measurement value out of range: {result}"
        
        return True, "Lightning: ✓ Installed and functional"
    
    except ImportError:
        return False, "Lightning: ✗ Not installed (pip install pennylane-lightning)"
    except Exception as e:
        return False, f"Lightning: ✗ Error during execution: {str(e)}"


def check_all_backends() -> Dict[str, Tuple[bool, str]]:
    """Check all available quantum backends.

    Returns:
        Dictionary mapping backend names to (success, message) tuples.
    """
    results = {
        "pennylane": check_pennylane(),
        "lightning": check_lightning(),
    }
    return results


def print_backend_status(results: Dict[str, Tuple[bool, str]]) -> None:
    """Print formatted backend status report.

    Args:
        results: Dictionary of backend check results.
    """
    print("\n" + "=" * 60)
    print("Quantum Backend Verification Report")
    print("=" * 60 + "\n")
    
    for backend, (success, message) in results.items():
        status_symbol = "✓" if success else "✗"
        print(f"{status_symbol} {message}")
    
    print("\n" + "=" * 60)
    
    # Summary
    total = len(results)
    passed = sum(1 for success, _ in results.values() if success)
    print(f"Summary: {passed}/{total} backends functional")
    print("=" * 60 + "\n")


def main() -> int:
    """Run backend verification and return exit code.

    Returns:
        0 if all backends are functional, 1 otherwise.
    """
    results = check_all_backends()
    print_backend_status(results)
    
    # Return 0 if at least PennyLane (the primary backend) works
    pennylane_ok = results.get("pennylane", (False, ""))[0]
    
    if not pennylane_ok:
        print("ERROR: PennyLane (primary backend) is not functional.")
        print("Please install it with: pip install pennylane")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
