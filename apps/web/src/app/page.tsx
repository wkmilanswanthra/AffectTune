"use client";
import { useEffect, useRef, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const EMOTIONS = [
  "happy",
  "sad",
  "angry",
  "fear",
  "surprise",
  "neutral",
  "disgust",
];

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [wFace, setWFace] = useState(0.6);
  const [wVoice, setWVoice] = useState(0.4);

  const [probs, setProbs] = useState<number[]>([0, 0, 0, 0, 0, 1, 0]);
  const [va, setVA] = useState<{ val: number; aro: number }>({
    val: 0,
    aro: 0,
  });

  useEffect(() => {
    let stream: MediaStream | null = null;
    let inFlight = false;
    let cancelled = false;
    let timer: number | null = null;

    const start = async () => {
      console.log("Requesting media...");
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          if (videoRef.current.readyState < 2) {
            await new Promise<void>((res) => {
              const onLoaded = () => {
                videoRef.current?.removeEventListener(
                  "loadedmetadata",
                  onLoaded
                );
                res();
              };
              videoRef.current?.addEventListener("loadedmetadata", onLoaded);
            });
          }
        }
        const tick = async () => {
          console.log("Tick");
          if (cancelled || inFlight) {
            timer = window.setTimeout(tick, 800);
            return;
          }
          if (!videoRef.current || !canvasRef.current) {
            timer = window.setTimeout(tick, 800);
            return;
          }
          const v = videoRef.current;
          const c = canvasRef.current;

          const vw = v.videoWidth || 0;
          const vh = v.videoHeight || 0;
          if (vw === 0 || vh === 0) {
            timer = window.setTimeout(tick, 800);
            return;
          }

          c.width = vw;
          c.height = vh;

          const g = c.getContext("2d");
          if (!g) {
            timer = window.setTimeout(tick, 800);
            return;
          }

          g.drawImage(v, 0, 0, c.width, c.height);

          const blob = await new Promise<Blob | null>((res) =>
            c.toBlob((b) => res(b), "image/jpeg", 0.7)
          );
          if (!blob) {
            timer = window.setTimeout(tick, 800);
            return;
          }

          const form = new FormData();
          form.append("file", blob, "frame.jpg");

          inFlight = true;
          try {
            const r = await fetch("http://127.0.0.1:8000/infer/face", {
              method: "POST",
              body: form,
            });
            console.log("fetch returned", r.status);
            if (!r.ok) {
              const txt = await r.text().catch(() => "");
              throw new Error(`HTTP ${r.status} ${r.statusText} ${txt}`);
            }
            const face = await r.json();
            if (!cancelled) {
              if (Array.isArray(face.probs)) setProbs(face.probs);
              if (
                typeof face.valence === "number" &&
                typeof face.arousal === "number"
              ) {
                setVA({ val: face.valence, aro: face.arousal });
              }
            }
          } catch (e) {
            console.error(e);
          } finally {
            inFlight = false;
            if (!cancelled) timer = window.setTimeout(tick, 800);
          }
        };

        timer = window.setTimeout(tick, 800);
      } catch (err) {
        console.error("getUserMedia failed:", err);
      }
    };

    start();

    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const data = EMOTIONS.map((name, i) => ({ name, p: probs[i] ?? 0 }));

  return (
    <main className="p-6 grid gap-4 md:grid-cols-2">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold">Emotion MVP</h1>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full max-w-[640px] rounded border"
        />
        <canvas ref={canvasRef} className="hidden" />
        <div className="text-sm">
          Valence: {va.val.toFixed(2)} • Arousal: {va.aro.toFixed(2)}
        </div>
        <div className="space-y-2">
          <label className="block text-sm">
            Face weight: {wFace.toFixed(2)}
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={wFace}
            onChange={(e) => {
              const v = Number(e.target.value);
              setWFace(v);
              setWVoice(1 - v);
            }}
          />
          <label className="block text-sm">
            Voice weight: {wVoice.toFixed(2)}
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={wVoice}
            onChange={(e) => {
              const v = Number(e.target.value);
              setWVoice(v);
              setWFace(1 - v);
            }}
          />
        </div>
      </section>
      <section className="h-[360px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="p" />
          </BarChart>
        </ResponsiveContainer>
      </section>
    </main>
  );
}
