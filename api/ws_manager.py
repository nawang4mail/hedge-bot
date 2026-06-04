"""
WebSocket connection manager.
Maintains a registry of {channel_id → [WebSocket, ...]} for fan-out broadcast.
"""
from __future__ import annotations
import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self._channels: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, channel: str, ws: WebSocket):
        await ws.accept()
        self._channels[channel].append(ws)

    def disconnect(self, channel: str, ws: WebSocket):
        self._channels[channel] = [c for c in self._channels[channel] if c is not ws]

    async def broadcast(self, channel: str, data: dict[str, Any]):
        """Send to all subscribers on this channel; silently drop closed sockets."""
        dead = []
        for ws in self._channels.get(channel, []):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(channel, ws)
