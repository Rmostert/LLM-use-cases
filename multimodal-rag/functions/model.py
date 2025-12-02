import boto3
import json

# Generating Multimodal Embeddings using Amazon Titan Multimodal Embeddings model
def generate_multimodal_embeddings(model_id='amazon.titan-embed-image-v1', prompt=None, image=None, output_embedding_length=384):
    """
    Invoke the Amazon Titan Multimodal Embeddings model using Amazon Bedrock runtime.

    Args:
        prompt (str): The text prompt to provide to the model.
        image (str): A base64-encoded image data.
    Returns:
        str: The model's response embedding.
    """
    if not prompt and not image:
        raise ValueError("Please provide either a text prompt, base64 image, or both as input")
    
    # Initialize the Amazon Bedrock runtime client
    client = boto3.client(service_name="bedrock-runtime")
    
    body = {"embeddingConfig": {"outputEmbeddingLength": output_embedding_length}}
    
    if prompt:
        body["inputText"] = prompt
    if image:
        body["inputImage"] = image

    try:
        response = client.invoke_model(
                                        modelId=model_id,
                                        body=json.dumps(body),
                                        accept="application/json",
                                        contentType="application/json"
                                    )

        # Process and return the response
        result = json.loads(response.get("body").read())
        return result.get("embedding")

    except ClientError as err:
        print(f"Couldn't invoke Titan embedding model. Error: {err.response['Error']['Message']}")
        return None


# Generating RAG response with Amazon Nova
def invoke_nova_multimodal(model_id="amazon.nova-pro-v1:0", prompt=None, matched_items=None):
    """
    Invoke the Amazon Nova model.
    """


    # Define your system prompt(s).
    system_msg = [
                        { "text": """You are a helpful assistant for question answering. 
                                    The text context is relevant information retrieved. 
                                    The provided image(s) are relevant information retrieved."""}
                 ]

    # Define one or more messages using the "user" and "assistant" roles.
    message_content = []

    for item in matched_items:
        if item['type'] == 'text' or item['type'] == 'table':
            message_content.append({"text": item['text']})
        else:
            message_content.append({"image": {
                                                "format": "png",
                                                "source": {"bytes": item['image']},
                                            }
                                    })


    # Configure the inference parameters.
    inf_params = {"maxTokens": 500, 
                "topP": 0.9, 
                }

    # Define the final message list
    message_list = [
        {"role": "user", "content": message_content}
    ]
    
    # Adding the prompt to the message list
    message_list.append({"role": "user", "content": [{"text": prompt}]})

    native_request = {
        "messages": message_list,
        "system": system_msg,
        "inferenceConfig": inf_params,
    }

    # Initialize the Amazon Bedrock runtime client
    
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Invoke the model and extract the response body.

    response = client.converse(
        modelId=model_id,
        **native_request,
    )

    model_response = response["output"]["message"]["content"][0]["text"]
    
    return model_response