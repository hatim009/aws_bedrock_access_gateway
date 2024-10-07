from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http.response import StreamingHttpResponse
from django.conf import settings
from .streaming_utils import streamed_response


@api_view(['GET'])
def models(request):
    return Response([
        {
            "name": "Gandhi AI", 
            "id": "gandhi-ai-v1:0"
        }
    ])


@api_view(['POST'])
def chat(request):
    return StreamingHttpResponse(streamed_response(request), content_type='text/event-stream')
