import time
import json
import uuid
import traceback
from botocore.eventstream import EventStream
from django.conf import settings


from .gandhi_ai_rag import get_gandhi_ai_rag_response

def stream_response_to_bytes(response):
    if response:
        # to populate other fields when using exclude_unset=True
        response['system_fingerprint'] = "fp"
        response['object'] = "chat.completion.chunk"
        response['created'] = int(time.time())
        return "data: {}\n\n".format(json.dumps(response)).encode("utf-8")
    return "data: [DONE]\n\n".encode("utf-8")

def convert_finish_reason(finish_reason):
    """
    Below is a list of finish reason according to OpenAI doc:

    - stop: if the model hit a natural stop point or a provided stop sequence,
    - length: if the maximum number of tokens specified in the request was reached,
    - content_filter: if content was omitted due to a flag from our content filters,
    - tool_calls: if the model called a tool
    """
    if finish_reason:
        finish_reason_mapping = {
            "tool_use": "tool_calls",
            "finished": "stop",
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "complete": "stop",
            "content_filtered": "content_filter"
        }
        return finish_reason_mapping.get(finish_reason.lower(), finish_reason.lower())
    return None


def create_response_stream(model_id, message_id, chunk):
    """Parsing the Bedrock stream response chunk.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
    """
    if settings.DEBUG:
        # print("Bedrock response chunk: " + str(chunk))
        pass
    finish_reason = None
    message = None
    usage = None
    if "messageStart" in chunk:
        message = {
            'role': chunk["messageStart"]["role"],
            'content': "",
        }
    if "contentBlockStart" in chunk:
        # tool call start
        delta = chunk['contentBlockStart']['start']
        if "toolUse" in delta:
            # first index is content
            index = chunk["contentBlockStart"]["contentBlockIndex"] - 1
            message = {
                'tool_calls': [
                    {
                        'index': index,
                        'type': 'function',
                        'id': delta["toolUse"]["toolUseId"],
                        'function': {
                            'name':delta["toolUse"]["name"],
                            'arguments': '',
                        }
                    }
                ]
            }
    if "contentBlockDelta" in chunk:
        delta = chunk["contentBlockDelta"]["delta"]
        if "text" in delta:
            # stream content
            message = {
                'content': delta["text"],
            }
        else:
            # tool use
            index = chunk["contentBlockDelta"]["contentBlockIndex"] - 1
            message = {
                'tool_calls': [
                    {
                        'index': index,
                        'function': {
                            'arguments': delta["toolUse"]["input"],
                        }   
                    }
                ]
            }
    if "messageStop" in chunk:
        message = {}
        finish_reason = chunk["messageStop"]["stopReason"]

    if "metadata" in chunk:
        # usage information in metadata.
        metadata = chunk["metadata"]
        if "usage" in metadata:
            # token usage
            return {
                'id': message_id,
                'model': model_id,
                'choices': [],
                'usage': {
                    'prompt_tokens': metadata["usage"]["inputTokens"],
                    'completion_tokens': metadata["usage"]["outputTokens"],
                    'total_tokens': metadata["usage"]["totalTokens"],
                },
            }
    if message:
        return {
            'id': message_id,
            'model': model_id,
            'choices': [
                {
                    'index': 0,
                    'delta': message,
                    'logprobs': None,
                    'finish_reason': convert_finish_reason(finish_reason),
                }
            ],
            'usage': usage,
        }

    return None


def streamed_response(request):

    try:
        response = get_gandhi_ai_rag_response(request)
    except Exception as e:
        traceback.print_exc()
        response = {
            'stream': [
                {'messageStart': {'role': 'assistant'}},
                {'contentBlockDelta': {'delta': {'text': 'An '}, 'contentBlockIndex': 0}},
                {'contentBlockDelta': {'delta': {'text': 'issue '}, 'contentBlockIndex': 1}},
                {'contentBlockDelta': {'delta': {'text': 'occurred '}, 'contentBlockIndex': 2}},
                {'contentBlockDelta': {'delta': {'text': 'while '}, 'contentBlockIndex': 3}},
                {'contentBlockDelta': {'delta': {'text': 'generating '}, 'contentBlockIndex': 4}},
                {'contentBlockDelta': {'delta': {'text': 'the '}, 'contentBlockIndex': 5}},
                {'contentBlockDelta': {'delta': {'text': 'response, '}, 'contentBlockIndex': 6}},
                {'contentBlockDelta': {'delta': {'text': 'Please '}, 'contentBlockIndex': 7}},
                {'contentBlockDelta': {'delta': {'text': 'ask a '}, 'contentBlockIndex': 8}},
                {'contentBlockDelta': {'delta': {'text': 'different '}, 'contentBlockIndex': 9}},
                {'contentBlockDelta': {'delta': {'text': 'question.'}, 'contentBlockIndex': 10}},
                {'contentBlockStop': {'contentBlockIndex': 11}},
                {'messageStop': {'stopReason': 'end_turn'}},
                {'metadata': {'usage': {'inputTokens': 100, 'outputTokens': 20, 'totalTokens': 120}, 'metrics': {'latencyMs': 0}}}
            ]
        }

    for chunk in response['stream']:
        stream_response = create_response_stream(request.data['model'], "chatcmpl-" + str(uuid.uuid4())[:8], chunk)
        if not stream_response:
            continue
        if settings.DEBUG:
            # print("Proxy response :" + json.dumps(stream_response))
            pass
        if stream_response.get('choices'):
            yield stream_response_to_bytes(stream_response)
        elif request.data.get('stream_options') and request.data['stream_options'].get('include_usage'):
            # An empty choices for Usage as per OpenAI doc below:
            # if you set stream_options: {"include_usage": true}.
            # an additional chunk will be streamed before the data: [DONE] message.
            # The usage field on this chunk shows the token usage statistics for the entire request,
            # and the choices field will always be an empty array.
            # All other chunks will also include a usage field, but with a null value.
            yield stream_response_to_bytes(stream_response)

    # return an [DONE] message at the end.
    yield stream_response_to_bytes(None)