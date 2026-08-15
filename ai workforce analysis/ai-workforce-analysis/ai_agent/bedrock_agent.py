import boto3
import os

REGION = "eu-north-1"
MODEL_ID = "amazon.nova-lite-v1:0"

def get_bedrock_client():
    try:
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=REGION
        )
    except Exception:
        return None

def generate_local_hr_recommendation(employee_data, risk_category, risk_score=None):
    emp_id = employee_data.get("employeeid", "Employee")
    first_name = employee_data.get("firstname", "")
    last_name = employee_data.get("lastname", "")
    full_name = f"{first_name} {last_name}".strip() or f"Employee #{emp_id}"
    dept = employee_data.get("department_clean", employee_data.get("departmenttype", "Department"))
    sat = employee_data.get("satisfactionscore", "N/A")
    eng = employee_data.get("engagementscore", "N/A")
    tenure = employee_data.get("tenure_years", "N/A")
    
    try:
        score_val = float(risk_score) if risk_score is not None else None
        score_str = f"{score_val:.1%}" if score_val is not None else "N/A"
    except Exception:
        score_str = str(risk_score)
    
    return f"""### AI HR Risk Assessment for {full_name}

**Attrition Risk Level:** {risk_category} (Score: {score_str})
**Department:** {dept} | **Tenure:** {tenure} years | **Satisfaction:** {sat}/5 | **Engagement:** {eng}/5

#### 1. Risk Explanation:
The predictive attrition model categorizes this employee as **{risk_category}** with a risk score of **{score_str}**. Key indicators including workplace satisfaction ({sat}/5) and engagement rating ({eng}/5) provide critical signals for retention probability.

#### 2. Key Visible Data Factors:
- **Tenure:** {tenure} years within {dept}
- **Satisfaction Rating:** {sat} out of 5
- **Engagement Rating:** {eng} out of 5
- **Organizational Placement:** {dept} department

#### 3. Recommended HR Actions:
1. **Targeted 1-on-1 Check-in:** Schedule a dedicated retention conversation to discuss current responsibilities, career trajectory, and team dynamics.
2. **Growth & Upskilling Pathways:** Identify mentoring or advanced project opportunities that align with career progression.
3. **Compensation & Recognition Review:** Benchmark current package against market standards and recognize recent milestone contributions.
"""

def generate_local_hr_chat_answer(employee_data, risk_category, risk_score, user_question):
    emp_id = employee_data.get("employeeid", "Employee")
    first_name = employee_data.get("firstname", "")
    last_name = employee_data.get("lastname", "")
    full_name = f"{first_name} {last_name}".strip() or f"Employee #{emp_id}"
    dept = employee_data.get("department_clean", employee_data.get("departmenttype", "Department"))
    sat = employee_data.get("satisfactionscore", "N/A")
    eng = employee_data.get("engagementscore", "N/A")
    tenure = employee_data.get("tenure_years", "N/A")
    
    return f"""Based on the workforce analytics intelligence for **{full_name}** ({dept}):

- **Risk Category:** {risk_category} (Risk Score: {risk_score})
- **Employee Indicators:** Satisfaction Score: {sat}/5, Engagement Score: {eng}/5, Tenure: {tenure} years.

**Response to your inquiry ("{user_question}"):**
The employee is currently assessed as **{risk_category}**. With a satisfaction index of {sat}/5 and tenure of {tenure} years in {dept}, HR should focus on proactive engagement, transparent communication regarding advancement opportunities, and regular pulse checks to mitigate attrition risk."""

def get_hr_recommendation(employee_data, risk_category, risk_score=None):
    """
    Generate an HR explanation using Amazon Bedrock with robust fallback.
    """
    try:
        bedrock = get_bedrock_client()
        if bedrock:
            prompt = f"""
You are an AI HR Assistant for a Workforce Analytics platform.

The workforce analytics model has already assessed this employee.

Risk Category: {risk_category}
Risk Score: {risk_score if risk_score is not None else "Not available"}

Employee Information:
{employee_data}

Your job is to explain the existing model assessment to HR.

Provide:
1. A concise explanation of the employee's attrition risk.
2. The important employee factors visible in the supplied data.
3. Three practical HR actions appropriate for the risk level.

Important rules:
- Do not change or contradict the supplied risk category.
- Do not invent employee information.
- Treat the recommendation as decision support.
"""
            response = bedrock.converse(
                modelId=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt}
                        ]
                    }
                ],
                inferenceConfig={
                    "maxTokens": 700,
                    "temperature": 0.2
                }
            )
            return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        print(f"Bedrock unavailable ({e}), using local intelligence fallback.")

    return generate_local_hr_recommendation(employee_data, risk_category, risk_score)


def chat_with_hr_assistant(
    employee_data,
    risk_category,
    risk_score,
    user_question
):
    """
    Answer HR questions about the currently selected employee
    using Amazon Bedrock with fallback.
    """
    try:
        bedrock = get_bedrock_client()
        if bedrock:
            prompt = f"""
You are an AI HR Assistant inside a Workforce Analytics platform.

HR is asking a question about a selected employee.

Employee Information:
{employee_data}

Attrition Risk Category: {risk_category}
Attrition Risk Score: {risk_score}

HR Question:
{user_question}

Answer the HR question using only the employee information and
risk information provided above.
"""
            response = bedrock.converse(
                modelId=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt}
                        ]
                    }
                ],
                inferenceConfig={
                    "maxTokens": 700,
                    "temperature": 0.2
                }
            )
            return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        print(f"Bedrock unavailable ({e}), using local intelligence fallback.")

    return generate_local_hr_chat_answer(
        employee_data, risk_category, risk_score, user_question
    )
