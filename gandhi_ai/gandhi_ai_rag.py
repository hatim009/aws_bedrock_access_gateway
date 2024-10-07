from django.conf import settings


def get_gandhi_ai_rag_response(request):
    messages = request.data['messages']

    return settings.BEDROCK_CLIENT.converse_stream(
        modelId=settings.LLAMA_MODEL_ID,
        messages=[
            {
                'role': message['role'], 
                'content': [
                    {
                        'text': message['content']
                    }
                ]
            }
            for message in messages
        ]
    )