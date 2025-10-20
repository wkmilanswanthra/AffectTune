export function openEcho(onMsg: (m: string) => void) {
  const ws = new WebSocket("ws://localhost:8000/ws/echo");
  ws.onmessage = (ev) => onMsg(ev.data as string);
  return ws;
}
