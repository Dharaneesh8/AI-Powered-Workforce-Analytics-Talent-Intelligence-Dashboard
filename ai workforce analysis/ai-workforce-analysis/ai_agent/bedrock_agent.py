import boto3

REGION = "eu-north-1"
MODEL_ID = "amazon.nova-lite-v1:0"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION
)


def get_hr_recommendation(employee_data, risk_category, risk_score=None):
    """
    Generate an HR explanation using Amazon Bedrock.

    risk_category:
        High Risk
        Medium Risk
        Low Risk
    """

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
- Do not assume compensation problems, manager problems, personal issues,
  or other causes unless those facts are supplied.
- Clearly distinguish facts from possible interpretations.
- Treat the recommendation as decision support, not a definitive statement
  about whether an employee will leave.
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


def chat_with_hr_assistant(
    employee_data,
    risk_category,
    risk_score,
    user_question
):
    """
    Answer HR questions about the currently selected employee
    using Amazon Bedrock.
    """

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

Rules:
- Do not change or contradict the supplied risk category.
- Do not invent employee facts.
- Do not claim that the employee will definitely leave or stay.
- If the provided data is insufficient to answer something, say that.
- Clearly distinguish observed employee data from possible interpretations.
- Give practical HR guidance when appropriate.
- Keep the response clear and useful for an HR professional.
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
