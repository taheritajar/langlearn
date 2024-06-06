import os
from banner import print_banner
from restapi.app import app
# from fastapi import FastAPI
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




if __name__ == "__main__":



    # Call the function to print the banner
    print_banner()
    uvicorn.run(app, host="0.0.0.0", port=8000)
