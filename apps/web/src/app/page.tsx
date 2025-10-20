"use client";
import { useEffect, useRef, useState } from "react";
import { openEcho } from "@/lib/ws";
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
  const [audioAllowed, setAudioAllowed] = useState<boolean | null>(false);
  const [echo, setEcho] = useState<string>("(none)");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [probs, setProbs] = useState<number[]>([0, 0, 0, 0, 0, 1, 0]);
  const [va, setVA] = useState<{ val: number; aro: number }>({
    val: 0,
    aro: 0,
  });

  useEffect(() => {
    (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      if (videoRef.current) videoRef.current.srcObject = stream;
      setAudioAllowed(true);

      const tick = async () => {
        if (!videoRef.current || !canvasRef.current) return;
        const v = videoRef.current;
        const c = canvasRef.current;
        c.width = v.videoWidth;
        c.height = v.videoHeight;
        const g = c.getContext("2d");
        if (!g) return;
        g.drawImage(v, 0, 0, c.width, c.height);
        const blob: Blob = await new Promise((res) =>
          c.toBlob((b) => res(b!), "image/jpeg", 0.7)
        );
        const form = new FormData();
        form.append("file", blob, "frame.jpg");
        const face = await fetch("http://localhost:8000/infer/face", {
          method: "POST",
          body: form,
        }).then((r) => r.json());
        setProbs(face.probs);
        setVA({ val: face.valence, aro: face.arousal });
      };

      const h = setInterval(tick, 800);
      return () => clearInterval(h);
    })().catch(console.error);

    const ws = openEcho((m) => setEcho(m));
    const t = setInterval(() => {
      try {
        ws.send("ping");
      } catch {}
    }, 2000);
    return () => {
      clearInterval(t);
      ws.close();
    };
  }, []);

  const data = EMOTIONS.map((e, i) => ({ name: e, p: probs[i] }));

  return (
    <main className="p-6 grid gap-4 md:grid-cols-2">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold">Emotion MVP</h1>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full max-w-[640px] rounded border"
        />
        <canvas ref={canvasRef} className="hidden" />
        <div className="text-sm">
          Valence: {va.val.toFixed(2)} • Arousal: {va.aro.toFixed(2)}
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
