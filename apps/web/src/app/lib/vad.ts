export function simpleVAD(samples: Float32Array, threshold = 0.02): boolean {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += Math.abs(samples[i]);
  return sum / samples.length > threshold;
}
