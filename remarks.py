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
     "Antenatal_Peripartum_Maternal_Complications": "{predictions.get('Antenatal_Peripartum_Maternal_Complications', 'No complication')}",
     "Neonatal__Fetal_Complications": "{predictions.get('Neonatal__Fetal_Complications', 'No complication')}"
   }}
 }}

   You are a medical assistant for community midwives. 
Your task is to generate short, clear instructions (2–3 lines) based on the predictions given.

Rules:
1. If "Antenatal_Peripartum_Maternal_Complications" is one of 
   [Massive Blood Transfusion, Postpartum Hemorrhage, Preeclampsia, Preterm Labour,
   Acute Pyelonephritis, Acute Renal Failure, ICU Admission, Eclampsia, Sudden Maternal Collapse,
   Placental Abruption, Placental Abruption and ICU Admission, Cardiac Failure, Pulmonary Edema,
   Ectopic Pregnancy, Hepatitis E and Termination of Pregnancy and Postpartum Hemorrhage,
   Hepatitis E and Termination of Pregnancy and Postpartum Hemorrhage and Shock, Imminent Eclampsia,
   Neglected Transverse lie, Pulmonary Embolism, Shock, Peripartum Hysterectomy, Ruptured Uterus,
   Scar Dehiscence, Sepsis], 
   always write: "These prediction values need urgent referral to nearest healthcare facility."

2. If "Neonatal__Fetal_Complications" is one of 
   [Miscarriage, Neonatal death, Birth asphyxia, Neonatal sepsis, Congenital anomalies, 
   Congenital heart defects, Neural tube defects, Chromosomal abnormalities,
   Neonatal meningitis, Neonatal pneumonia, Severe respiratory distress syndrome,
   Severe jaundice requiring exchange transfusion, Neonatal seizures, Neonatal hypoglycemia], 
   always write: "These prediction values need urgent referral to nearest healthcare facility."

3. If "Antenatal_Peripartum_Maternal_Complications" is one of 
   [Anemia, Rh Incompatibility and antiD prophylaxis, Gestational Diabetes Mellitus, Obstetric Cholestasis,
   Urinary Tract Infection, Cervical Incompetence, Chorioamnionitis, Cord Prolapse,
   Deep Vein Thrombosis, Diabetes Mellitus, Obstructed Labour, Perineal Tears,
   Acute Hepatitis E, Hypertension, Hypertension AND PPROM, Intrauterine Growth Restriction,
   Low Lying Placenta, Molar Pregnancy, Puerperal Pyrexia, ECV,
   Placenta Accreta Spectrum, PPROM, Recurrent Urinary Tract Infection,
   Shoulder Dystocia, Thalasemia Minor, Threatened Miscarriage, Threatened Preterm Labour,
   Thyroid Disease, Uterine Perforation], 
   always write: "These prediction values indicate a high risk pregnancy which needs to be managed closely in liaison with Expert Gynecologist."

4. If all three conditions are met for LOW RISK PREGNANCY:
   - "Antenatal_Peripartum_Maternal_Complications" = "No Complication" 
   - "Mode_of_delivery" = "Not delivered yet" OR "Vaginal"
   - "Neonatal__Fetal_Complications" = "No Complication" OR "Normal"
   Then write: "These prediction values indicate currently low risk pregnancy to be followed for routine followup visits."

5. If "Mode_of_delivery" is one of 
   [Induced Abortion, Laparotomy, Laparotomy and Peripartum Hysterectomy and ICU Admission,
   Laparotomy and Repair of Uterus], 
   always write: "These prediction values need urgent referral to nearest healthcare facility."

6. If no serious complication is detected, then give guidance according to "Mode_of_delivery":
   - If "Mode_of_delivery" = "Not delivered yet" → "Patient not delivered yet. Continue regular monitoring and follow antenatal care schedule."
   - If "Mode_of_delivery" = "Normal Delivery" OR "Vaginal" → "Normal delivery predicted. Continue routine care and prepare for safe delivery."
   - If "Mode_of_delivery" = "Cesarean Section" → "Cesarean delivery predicted. Ensure hospital preparation and skilled assistance."
   - If "Mode_of_delivery" = "Assisted Delivery" OR "Assisted Vaginal Delivery" → "Assisted delivery predicted. Ensure skilled assistance and follow safety measures."
   - If "Mode_of_delivery" = "ERPC" OR "Hysterotomy" OR "Termination of Pregnancy" OR "Medical Termination of Pregnancy and Vaginal" OR "Methotrexate" OR "Suction Evacuation" → "Specialized procedure predicted. Ensure hospital preparation and expert gynecological care."
   - If "Mode_of_delivery" = "ECV and Vaginal" OR "Induction of Labour" OR "Induction of Labour and Vaginal" → "Induction procedure predicted. Monitor closely and prepare for assisted delivery."
   - If "Mode_of_delivery" = "Laparoscopy" → "Laparoscopic procedure predicted. Ensure specialized surgical preparation."
   - If "Mode_of_delivery" = "Not Delivered Yet and Cervical Cerclage" OR "Not Delivered Yet and ECV" → "Specialized antenatal procedure predicted. Continue close monitoring with expert care."

Output Format:
{{
   "predictions": {{
     "Mode_of_delivery": "<value>",
     "Antenatal_Peripartum_Maternal_Complications": "<value>",
     "Neonatal__Fetal_Complications": "<value>"
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
   [Neonatal death, Birth asphyxia, Neonatal sepsis, Congenital anomalies, 
   Congenital heart defects, Neural tube defects, Chromosomal abnormalities,
   Neonatal meningitis, Neonatal pneumonia, Severe respiratory distress syndrome,
   Severe jaundice requiring exchange transfusion, Neonatal seizures, Neonatal hypoglycemia], 
   always write: "These prediction values need urgent referral to nearest healthcare facility."

2. If "Neonatal_Fetal_Complications" is one of 
   [Respiratory distress syndrome, Jaundice, Low birth weight, Fetal distress,
   Intrauterine Growth Restriction, Preterm birth complications, Neonatal jaundice,
   Respiratory distress, Neonatal infection, Birth trauma, Premature birth,
   Premature birth and respiratory distress syndrome, Neonatal complications requiring NICU], 
   always write: "Neonatal complication predicted: <complication>. Immediate neonatal care required. Refer to specialized neonatal unit."

3. If "Neonatal_Fetal_Complications" = "No Complication" OR "Normal":
   Then write: "No neonatal complications predicted. Continue routine neonatal monitoring and care."

4. For other minor complications:
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
   [Cardiac Failure, Shock, ICU Admission, Acute Hepatic Failure, Acute Renal Failure,
   Sepsis, Deep Vein Thrombosis and Pulmonary Embolism, Deep Vein Thrombosis and Sepsis,
   Eclampsia and ICU Admission, Infected Wound and sepsis, Intracranial Hemorrhage and ICU Admission,
   Laparotomy and Shock and Acute Renal Failure and ICU Admission, Laparotomy and Shock and ICU Admission,
   Peripartum Cardiomyopathy, Postpartum Hemorrhage and Sepsis, Postpartum Hemorrhage and ICU Admission,
   Postpartum Hemorrhage and Shock and ICU Admission, Preeclampsia, Sudden Maternal Collapse], 
   always write: "These prediction values need urgent referral to nearest healthcare facility."

2. If "Postnatal_Maternal_Complications" is one of 
   [Postpartum depression, Uterine infection, Wound Hematoma, Acute Hepatitis E, Acute Mastitis, 
   Anemia, Hypertension, Infected Surgical/Episiotomy wound, Postpartum Eclampsia, RPOCs, 
   Urinary Tract Infection], 
   always write: "Postnatal complication predicted: <complication>. Immediate medical attention required. Refer to hospital."

3. If "Postnatal_Maternal_Complications" = "No Complication" OR "Normal Findings":
   Then write: "No postnatal complications predicted. Continue routine postnatal care and monitoring."

4. For other minor complications:
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
        if endpoint == "neonatal":
            return generate_neonatal_fallback(predictions)
        elif endpoint == "postnatal":
            return generate_postnatal_fallback(predictions)
        else:
            return generate_antenatal_fallback(predictions)
    except Exception as e:
        logger.error(f"Error in fallback medical instruction generation: {str(e)}")
        return "Unable to generate specific instruction. Please consult with senior medical staff."

def generate_neonatal_fallback(predictions: Dict[str, str]) -> str:
    """
    Fallback for neonatal complications
    """
    try:
        neonatal_complications = predictions.get("Neonatal_Fetal_Complications", "No complication")
        
        # Critical neonatal complications
        critical_neonatal = [
            "Neonatal death", "Birth asphyxia", "Neonatal sepsis", "Congenital anomalies", 
            "Congenital heart defects", "Neural tube defects", "Chromosomal abnormalities",
            "Neonatal meningitis", "Neonatal pneumonia", "Severe respiratory distress syndrome",
            "Severe jaundice requiring exchange transfusion", "Neonatal seizures", "Neonatal hypoglycemia"
        ]
        
        # High risk neonatal complications
        high_risk_neonatal = [
            "Respiratory distress syndrome", "Jaundice", "Low birth weight", "Fetal distress",
            "Intrauterine Growth Restriction", "Preterm birth complications", "Neonatal jaundice",
            "Respiratory distress", "Neonatal infection", "Birth trauma", "Premature birth",
            "Premature birth and respiratory distress syndrome", "Neonatal complications requiring NICU"
        ]
        
        if neonatal_complications in critical_neonatal:
            return "These prediction values need urgent referral to nearest healthcare facility."
        elif neonatal_complications in high_risk_neonatal:
            return "Neonatal complication predicted: " + neonatal_complications + ". Immediate neonatal care required. Refer to specialized neonatal unit."
        elif neonatal_complications.lower() in ["no complication", "normal"]:
            return "No neonatal complications predicted. Continue routine neonatal monitoring and care."
        else:
            return "Minor neonatal condition predicted: " + neonatal_complications + ". Monitor closely and follow standard neonatal care protocols."
            
    except Exception as e:
        logger.error(f"Error in neonatal fallback: {str(e)}")
        return "Unable to generate specific instruction. Please consult with senior medical staff."

def generate_postnatal_fallback(predictions: Dict[str, str]) -> str:
    """
    Fallback for postnatal complications
    """
    try:
        postnatal_complications = predictions.get("Postnatal_Maternal_Complications", "No Complication")
        
        # Critical postnatal complications
        critical_postnatal = [
            "Cardiac Failure", "Shock", "ICU Admission", "Acute Hepatic Failure", "Acute Renal Failure",
            "Sepsis", "Deep Vein Thrombosis and Pulmonary Embolism", "Deep Vein Thrombosis and Sepsis",
            "Eclampsia and ICU Admission", "Infected Wound and sepsis", "Intracranial Hemorrhage and ICU Admission",
            "Laparotomy and Shock and Acute Renal Failure and ICU Admission", "Laparotomy and Shock and ICU Admission",
            "Peripartum Cardiomyopathy", "Postpartum Hemorrhage and Sepsis", "Postpartum Hemorrhage and ICU Admission",
            "Postpartum Hemorrhage and Shock and ICU Admission", "Preeclampsia", "Sudden Maternal Collapse"
        ]
        
        # High risk postnatal complications
        high_risk_postnatal = [
            "Postpartum depression", "Uterine infection", "Wound Hematoma", "Acute Hepatitis E", "Acute Mastitis", 
            "Anemia", "Hypertension", "Infected Surgical/Episiotomy wound", "Postpartum Eclampsia", "RPOCs", 
            "Urinary Tract Infection"
        ]
        
        if postnatal_complications in critical_postnatal:
            return "These prediction values need urgent referral to nearest healthcare facility."
        elif postnatal_complications in high_risk_postnatal:
            return "Postnatal complication predicted: " + postnatal_complications + ". Immediate medical attention required. Refer to hospital."
        elif postnatal_complications.lower() in ["no complication", "normal findings"]:
            return "No postnatal complications predicted. Continue routine postnatal care and monitoring."
        else:
            return "Minor postnatal condition predicted: " + postnatal_complications + ". Monitor closely and follow standard postnatal care protocols."
            
    except Exception as e:
        logger.error(f"Error in postnatal fallback: {str(e)}")
        return "Unable to generate specific instruction. Please consult with senior medical staff."

def generate_antenatal_fallback(predictions: Dict[str, str]) -> str:
    """
    Fallback for antenatal complications
    """
    try:
        mode_of_delivery = predictions.get("Mode_of_delivery", "Not delivered yet")
        complications = predictions.get("Antenatal_Peripartum_Maternal_Complications", "No complication")
        neonatal_complications = predictions.get("Neonatal__Fetal_Complications", "No complication")
        
        # List of critical complications that need urgent referral
        critical_complications = [
            "Massive Blood Transfusion", "Postpartum Hemorrhage", "Preeclampsia", "Preterm Labour",
            "Acute Pyelonephritis", "Acute Renal Failure", "ICU Admission", "Eclampsia", "Sudden Maternal Collapse",
            "Placental Abruption", "Placental Abruption and ICU Admission", "Cardiac Failure", "Pulmonary Edema",
            "Ectopic Pregnancy", "Hepatitis E and Termination of Pregnancy and Postpartum Hemorrhage",
            "Hepatitis E and Termination of Pregnancy and Postpartum Hemorrhage and Shock", "Imminent Eclampsia",
            "Neglected Transverse lie", "Pulmonary Embolism", "Shock", "Peripartum Hysterectomy", "Ruptured Uterus",
            "Scar Dehiscence", "Sepsis"
        ]
        
        # List of critical neonatal complications that need urgent referral
        critical_neonatal = [
            "Miscarriage", "Neonatal death", "Birth asphyxia", "Neonatal sepsis", "Congenital anomalies", 
            "Congenital heart defects", "Neural tube defects", "Chromosomal abnormalities",
            "Neonatal meningitis", "Neonatal pneumonia", "Severe respiratory distress syndrome",
            "Severe jaundice requiring exchange transfusion", "Neonatal seizures", "Neonatal hypoglycemia"
        ]
        
        # List of high risk complications that need expert gynecologist
        high_risk_complications = [
            "Anemia", "Rh Incompatibility and antiD prophylaxis", "Gestational Diabetes Mellitus", "Obstetric Cholestasis",
            "Urinary Tract Infection", "Cervical Incompetence", "Chorioamnionitis", "Cord Prolapse",
            "Deep Vein Thrombosis", "Diabetes Mellitus", "Obstructed Labour", "Perineal Tears",
            "Acute Hepatitis E", "Hypertension", "Hypertension AND PPROM", "Intrauterine Growth Restriction",
            "Low Lying Placenta", "Miscarriage", "Molar Pregnancy", "Puerperal Pyrexia", "ECV",
            "Placenta Accreta Spectrum", "PPROM", "Recurrent Urinary Tract Infection",
            "Shoulder Dystocia", "Thalasemia Minor", "Threatened Miscarriage", "Threatened Preterm Labour",
            "Thyroid Disease", "Uterine Perforation"
        ]
        
        # Check for critical complications
        if complications in critical_complications:
            return "These prediction values need urgent referral to nearest healthcare facility."
        
        # Check for critical neonatal complications
        if neonatal_complications in critical_neonatal:
            return "These prediction values need urgent referral to nearest healthcare facility."
        
        # Check for high risk complications
        if complications in high_risk_complications:
            return "These prediction values indicate a high risk pregnancy which needs to be managed closely in liaison with Expert Gynecologist."
        
        # Check for critical mode of delivery
        critical_delivery_modes = [
            "Induced Abortion", "Laparotomy", "Laparotomy and Peripartum Hysterectomy and ICU Admission",
            "Laparotomy and Repair of Uterus"
        ]
        
        if mode_of_delivery in critical_delivery_modes:
            return "These prediction values need urgent referral to nearest healthcare facility."
        
        # Check for LOW RISK PREGNANCY conditions
        is_low_risk = (
            complications.lower() in ["no complication", "none", "not specified"] and
            mode_of_delivery.lower() in ["not delivered yet", "vaginal", "normal delivery"] and
            neonatal_complications.lower() in ["no complication", "normal", "none", "not specified"]
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
            elif mode_of_delivery.lower() in ["assisted delivery", "assisted vaginal delivery"]:
                return "Assisted delivery predicted. Ensure skilled assistance and follow safety measures."
            elif mode_of_delivery.lower() in ["erpc", "hysterotomy", "termination of pregnancy", "medical termination of pregnancy and vaginal", "methotrexate", "suction evacuation"]:
                return "Specialized procedure predicted. Ensure hospital preparation and expert gynecological care."
            elif mode_of_delivery.lower() in ["ecv and vaginal", "induction of labour", "induction of labour and vaginal"]:
                return "Induction procedure predicted. Monitor closely and prepare for assisted delivery."
            elif mode_of_delivery.lower() in ["laparoscopy"]:
                return "Laparoscopic procedure predicted. Ensure specialized surgical preparation."
            elif mode_of_delivery.lower() in ["not delivered yet and cervical cerclage", "not delivered yet and ecv"]:
                return "Specialized antenatal procedure predicted. Continue close monitoring with expert care."
            else:
                return f"Mode of delivery: {mode_of_delivery}. Continue appropriate monitoring and care."
        else:
            # Minor complications or other conditions
            return f"Minor condition detected: {complications}. Monitor closely and follow standard care protocols."
            
    except Exception as e:
        logger.error(f"Error in antenatal fallback: {str(e)}")
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
