// Service worker that serves the page's own files with the headers that make
// it cross-origin isolated (COOP same-origin, COEP require-corp), which
// browsers require before onnxruntime-web may run on several threads. The
// static host cannot set these headers itself. Requests to other origins
// (the models, onnxruntime-web) pass through untouched; they are CORS
// requests and so allowed under require-corp.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (new URL(request.url).origin !== self.location.origin) return;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  event.respondWith((async () => {
    const response = await fetch(request);
    if (response.status === 0) return response;
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  })());
});
