import os
import sys

def main():
    # Set environment variables to prevent macOS crashes and protobuf issues
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Exec into streamlit
    os.execvp("streamlit", ["streamlit", "run", "app.py", "--server.headless", "true"] + sys.argv[1:])

if __name__ == "__main__":
    main()
