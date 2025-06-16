#!/usr/bin/env python3
"""Start the Portfolio Optimizer API server."""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "ui" / "backend"
sys.path.append(str(backend_dir))

if __name__ == "__main__":
    try:
        import uvicorn
        from portfolio_optimizer_api import app
        
        print("Starting Portfolio Optimizer API Server...")
        print("API will be available at: http://localhost:8001")
        print("API Documentation: http://localhost:8001/docs")
        print("Press Ctrl+C to stop the server")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""Start the Portfolio Optimizer API server."""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "ui" / "backend"
sys.path.append(str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    from portfolio_optimizer_api import app
    
    print("Starting Portfolio Optimizer API Server...")
    print("API will be available at: http://localhost:8001")
    print("API Documentation: http://localhost:8001/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
