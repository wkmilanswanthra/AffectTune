export async function grabFrame(video: HTMLVideoElement): Promise<ImageBitmap> {
  const frame = await createImageBitmap(video);
  return frame;
}
