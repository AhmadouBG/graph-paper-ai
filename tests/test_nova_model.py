import boto3
from botocore.exceptions import ClientError

# 1. Provide your AWS credentials directly
AWS_ACCESS_KEY = "YOUR_ACCESS_KEY"
AWS_SECRET_KEY = "YOUR_SECRET_KEY"
AWS_REGION = "us-east-1"  # Change to your target region

try:
    # 2. Initialize the Bedrock Runtime client
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    print("Connecting to Amazon Bedrock...")

    # 3. Send a simple test prompt to Amazon Nova Micro
    response = bedrock.converse(
        modelId="amazon.nova-micro-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": "Reply with exactly the word 'SUCCESS' if you can read this."}]
            }
        ]
    )

    # 4. Extract and print the response text
    output_text = response["output"]["message"]["content"][0]["text"]
    print(f"\nModel Response: {output_text}")

except ClientError as e:
    error_code = e.response["Error"]["Code"]
    error_message = e.response["Error"]["Message"]
    
    # 5. Catch Throttling or Permission errors immediately
    print(f"\n[ERROR] Request failed!")
    print(f"Code: {error_code}")
    print(f"Message: {error_message}")
    
    if error_code == "ThrottlingException":
        print("\n👉 Fix: Your daily token limit is still 0. You must request a quota increase in the Service Quotas console.")
    elif error_code == "AccessDeniedException":
        print("\n👉 Fix: Your IAM user keys lack the 'bedrock:InvokeModel' permission policy.")
