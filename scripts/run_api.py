# ============================================================
# RUN API — runner script
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "serving.attrition_api:app",
        host   = "127.0.0.1",
        port   = 8000,
        reload = False   # reload=True + logs/ writes on every /predict call
                         # causes the reloader to restart mid-request — disabled here.
    )
