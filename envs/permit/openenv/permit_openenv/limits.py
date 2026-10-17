"""Bound incoming HTTP bodies and WebSocket messages before framework JSON parsing."""
from typing import Any

MAX_REQUEST_BYTES = 65536


class RequestLimits:
    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            chunks, size = [], 0
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                body = message.get('body', b'')
                size += len(body)
                if size > MAX_REQUEST_BYTES:
                    await send({'type': 'http.response.start', 'status': 413,
                                'headers': [(b'content-type', b'application/json')]})
                    await send({'type': 'http.response.body', 'body': b'{"error":"request exceeds input limits"}'})
                    return
                chunks.append(body)
                if not message.get('more_body', False):
                    break
            pending = True
            async def limited_receive():
                nonlocal pending
                if pending:
                    pending = False
                    return {'type': 'http.request', 'body': b''.join(chunks), 'more_body': False}
                return await receive()
        elif scope['type'] == 'websocket':
            async def limited_receive():
                message = await receive()
                data = message.get('bytes') or (message.get('text') or '').encode()
                if len(data) > MAX_REQUEST_BYTES:
                    await send({'type': 'websocket.close', 'code': 1009, 'reason': 'request exceeds input limits'})
                    return {'type': 'websocket.disconnect', 'code': 1009}
                return message
        else:
            limited_receive = receive
        await self.app(scope, limited_receive, send)
