# remarks.py - FastAPI endpoint for generating medical remarks using Ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import json
from ollama import generate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Medical Remarks Generator", version="1.0.0")

class PredictionData(BaseModel):
    """Model for prediction data input"""
    predictions: Dict[str, str]
    endpoint: str

class RemarksResponse(BaseModel):
    """Model for remarks response"""
    predictions: Dict[str, str]
    instruction: str
    status: str

def create_medical_prompt(predictions: Dict[str, str], endpoint: str) -> str:
    """
    Create the medical prompt based on endpoint and predictions
    """
    
    if endpoint == "modeofdelivery_antinatal":
        # Prompt for Mode of Delivery + Antenatal Complications
        prompt = f""" {{
   "predictions": {{
     "Mode_of_delivery": "{predictions.get('Mode_of_delivery', 'Not delivered yet')}",
     "Antenatal_Peripartum_Maternal_Complications": "{predictions.get('Antenatal_Peripartum_Maternal_Complications', 'No complication')}"
   }}
 }}

   You are a medical assistant for community midwives. 
Your task is to generate short, clear instructions (2–3 lines) based on the predictions given.

Rules:
1. If "Antenatal_Peripartum_Maternal_Complications" is one of 
   [Acute renal failure, cardiac failure, Placenta previa, Antepartum haemorrhage, 
   Eclampsia, Pre-eclampsia, Placental abruption, Postpartum hemorrhage], 
   always write: "Complication predicted: <complication>. Please refer to Nearest Hospital."

2. If both conditions are met for LOW RISK PREGNANCY:
   - "Antenatal_Peripartum_Maternal_Complications" = "No Complication" 
   - "Mode_of_delivery" = "Not delivered yet" OR "Vaginal"
   Then write: "These prediction values indicate currently low risk pregnancy to be followed for routine followup visits."

3. If no serious complication is detected, then give guidance according to "Mode_of_delivery":
   - If "Mode_of_delivery" = "Not delivered yet" → "Patient not delivered yet. Continue regular monitoring and follow antenatal care schedule."
   - If "Mode_of_delivery" = "Normal Delivery" OR "Vaginal" → "Normal delivery predicted. Continue routine care and prepare for safe delivery."
   - If "Mode_of_delivery" = "Cesarean Section" → "Cesarean delivery predicted. Ensure hospital preparation and skilled assistance."
   - If "Mode_of_delivery" = "Assisted Delivery" → "Assisted delivery predicted. Ensure skilled assistance and follow safety measures."

Output Format:
{{
   "predictions": {{
     "Mode_of_delivery": "<value>",
     "Antenatal_Peripartum_Maternal_Complications": "<value>"
   }},
   "instruction": "<final instruction for midwife>"
 }}

     """
    
    elif endpoint == "neonatal":
        # Prompt for Neonatal Complications
        prompt = f""" {{
   "predictions": {{
     "Neonatal_Fetal_Complications": "{predictions.get('Neonatal_Fetal_Complications', 'No complication')}"
   }}
 }}

   You are a medical assistant for community midwives. 
Your task is to generate short, clear instructions (2–3 lines) based on the neonatal/fetal complications prediction.

Rules:
1. If "Neonatal_Fetal_Complications" is one of 
   [Birth asphyxia, Neonatal sepsis, Respiratory distress syndrome, Jaundice, 
   Low birth weight, Congenital anomalies, Neonatal death, Fetal distress], 
   always write: "Neonatal complication predicted: <complication>. Immediate neonatal care required. Refer to specialized neonatal unit."

2. If "Neonatal_Fetal_Complications" = "No Complication" OR "Normal":
   Then write: "No neonatal complications predicted. Continue routine neonatal monitoring and care."

3. For other minor complications:
   - Write: "Minor neonatal condition predicted: <complication>. Monitor closely and follow standard neonatal care protocols."

Output Format:
{{
   "predictions": {{
     "Neonatal_Fetal_Complications": "<value>"
   }},
   "instruction": "<final instruction for midwife>"
 }}

     """
    
    elif endpoint == "postnatal":
        # Prompt for Postnatal Complications
        prompt = f""" {{
   "predictions": {{
     "Postnatal_Maternal_Complications": "{predictions.get('Postnatal_Maternal_Complications', 'No Complication')}"
   }}
 }}

   You are a medical assistant for community midwives. 
Your task is to generate short, clear instructions (2–3 lines) based on the postnatal maternal complications prediction.

Rules:
1. If "Postnatal_Maternal_Complications" is one of 
   [Postpartum hemorrhage, Puerperal sepsis, Deep vein thrombosis, 
   Pulmonary embolism, Postpartum depression, Uterine infection], 
   always write: "Postnatal complication predicted: <complication>. Immediate medical attention required. Refer to hospital."

2. If "Postnatal_Maternal_Complications" = "No Complication" OR "Normal Findings":
   Then write: "No postnatal complications predicted. Continue routine postnatal care and monitoring."

3. For other minor complications:
   - Write: "Minor postnatal condition predicted: <complication>. Monitor closely and follow standard postnatal care protocols."

Output Format:
{{
   "predictions": {{
     "Postnatal_Maternal_Complications": "<value>"
   }},
   "instruction": "<final instruction for midwife>"
 }}

     """
    
    else:
        # Default prompt for unknown endpoints
        prompt = f""" {{
   "predictions": {predictions}
 }}

   You are a medical assistant for community midwives. 
Your task is to generate short, clear instructions (2–3 lines) based on the predictions given.

Please provide appropriate medical guidance based on the prediction values.

Output Format:
{{
   "predictions": {predictions},
   "instruction": "<final instruction for midwife>"
 }}

     """
    
    return prompt

def generate_medical_instruction_with_ollama(predictions: Dict[str, str], endpoint: str) -> str:
    """
    Generate medical instruction using Ollama model
    """
    try:
        # model can be: 'gemma3', 'llama3.2:1b', 'codellama', etc.
        model = "gemma3"
        
        # Create the prompt using the common function
        prompt = create_medical_prompt(predictions, endpoint)
        
        print(prompt)
        # Generate using Ollama
        resp = generate(model, prompt)
        
        # Extract the response
        response_text = resp.get("response") or str(resp)
        
        # Try to parse JSON response from Ollama
        try:
            # Look for JSON in the response
            if "{" in response_text and "}" in response_text:
                # Extract JSON part
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                json_str = response_text[start_idx:end_idx]
                
                parsed_response = json.loads(json_str)
                return parsed_response.get("instruction", response_text)
            else:
                return response_text
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return response_text
            
    except Exception as e:
        logger.error(f"Error generating medical instruction with Ollama: {str(e)}")
        # Fallback to rule-based approach if Ollama fails
        return generate_medical_instruction_fallback(predictions, endpoint)

def generate_medical_instruction_fallback(predictions: Dict[str, str], endpoint: str = "") -> str:
    """
    Fallback method to generate medical instruction based on predictions without using Ollama
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
        
        # Check for LOW RISK PREGNANCY conditions
        is_low_risk = (
            complications.lower() in ["no complication", "none", "not specified"] and
            mode_of_delivery.lower() in ["not delivered yet", "vaginal", "normal delivery"]
        )
        
        if is_low_risk:
            return "These prediction values indicate currently low risk pregnancy to be followed for routine followup visits."
        
        # If no serious complication, provide guidance based on mode of delivery
        if complications.lower() in ["no complication", "none", "not specified"]:
            if mode_of_delivery == "Not delivered yet":
                return "Patient not delivered yet. Continue regular monitoring and follow antenatal care schedule."
            elif mode_of_delivery.lower() in ["normal delivery", "vaginal"]:
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
        logger.error(f"Error in fallback medical instruction generation: {str(e)}")
        return "Unable to generate specific instruction. Please consult with senior medical staff."

@app.post("/generate-remarks", response_model=RemarksResponse)
async def generate_remarks(data: PredictionData):
    """
    Generate medical remarks and instructions based on prediction data using Ollama
    """
    try:
        # Validate input data
        if not data.predictions:
            raise HTTPException(status_code=400, detail="Predictions data is required")
        
        # Generate medical instruction using Ollama
        instruction = generate_medical_instruction_with_ollama(data.predictions,data.endpoint)
        
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
