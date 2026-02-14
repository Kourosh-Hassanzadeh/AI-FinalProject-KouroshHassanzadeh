import sys
from pathlib import Path
import streamlit as st


# Path Configuration

# Resolve absolute paths based on the location of this file (demo/main.py)
_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"

# Insert paths at the beginning of sys.path to prioritize local modules
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))


def main() -> None:
    """Initialize and run the Streamlit application.

    This function attempts to import the main application class and execute it.
    It catches and visually reports import errors or unexpected runtime
    exceptions directly within the Streamlit interface to aid debugging.
    """
    try:
        # Import is delayed until the module search paths are fully configured
        from app import ClassicalVisionApp
        
        # Instantiate and run the application
        vision_app = ClassicalVisionApp()
        vision_app.run()

    except ImportError as err:
        st.error("❌ Critical Import Error")
        st.code(f"Details: {err}")
        st.warning(
            "Please ensure that `app.py` is in the same directory as `main.py` "
            "and that the `src/` directory exists in the parent folder."
        )
        st.stop()
        
    except Exception as err:
        st.error(f"❌ An unexpected runtime error occurred: {err}")
        st.stop()


if __name__ == "__main__":
    main()