# remarks.py - FastAPI endpoint for generating medical remarks
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Medical Remarks Generator", version="1.0.0")

class PredictionData(BaseModel):
    """Model for prediction data input"""
    predictions: Dict[str, str]

class RemarksResponse(BaseModel):
    """Model for remarks response"""
    predictions: Dict[str, str]
    instruction: str
    status: str

def generate_medical_instruction(predictions: Dict[str, str]) -> str:
    """
    Generate medical instruction based on predictions without using Ollama
    """
    try:
        mode_of_delivery = predictions.get("Mode_of_delivery", "Not delivered yet")
        complications = predictions.get("Antenatal_Peripartum_Maternal_Complications", "No complication")
        
        # List of serious complications that require immediate hospital referral
        serious_complications = [
            "Acute renal failure", "cardiac failure", "Placenta previa", 
            "Antepartum haemorrhage", "Eclampsia", "Pre-eclampsia", 
            "Placental abruption", "Postpartum hemorrhage"
        ]
        
        # Check for serious complications
        if complications in serious_complications:
            return f"Complication predicted: {complications}. Please refer to Nearest Hospital."
        
        # If no serious complication, provide guidance based on mode of delivery
        if complications.lower() in ["no complication", "none", "not specified"]:
            if mode_of_delivery == "Not delivered yet":
                return "Patient not delivered yet. Continue regular monitoring and follow antenatal care schedule."
            elif mode_of_delivery == "Normal Delivery":
                return "Normal delivery predicted. Continue routine care and prepare for safe delivery."
            elif mode_of_delivery == "Cesarean Section":
                return "Cesarean delivery predicted. Ensure hospital preparation and skilled assistance."
            elif mode_of_delivery == "Assisted Delivery":
                return "Assisted delivery predicted. Ensure skilled assistance and follow safety measures."
            else:
                return f"Mode of delivery: {mode_of_delivery}. Continue appropriate monitoring and care."
        else:
            # Minor complications or other conditions
            return f"Minor condition detected: {complications}. Monitor closely and follow standard care protocols."
            
    except Exception as e:
        logger.error(f"Error generating medical instruction: {str(e)}")
        return "Unable to generate specific instruction. Please consult with senior medical staff."

@app.post("/generate-remarks", response_model=RemarksResponse)
async def generate_remarks(data: PredictionData):
    """
    Generate medical remarks and instructions based on prediction data
    """
    try:
        # Validate input data
        if not data.predictions:
            raise HTTPException(status_code=400, detail="Predictions data is required")
        
        # Generate medical instruction
        instruction = generate_medical_instruction(data.predictions)
        
        # Prepare response
        response = RemarksResponse(
            predictions=data.predictions,
            instruction=instruction,
            status="success"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in remarks generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Medical Remarks Generator",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Remarks Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate_remarks": "POST /generate-remarks",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("remarks:app", host="127.0.0.1", port=8002, reload=True)
