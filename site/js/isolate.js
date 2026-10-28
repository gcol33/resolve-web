// Puts the page under its service worker (coi.js), which makes it
// cross-origin isolated so the models can run on several threads. The first
// visit reloads once when the worker takes control; a browser that keeps the
// page unisolated under the worker, or has no service workers, predicts on
// one thread.
if (!self.crossOriginIsolated && "serviceWorker" in navigator) {
  const controlled = Boolean(navigator.serviceWorker.controller);
  navigator.serviceWorker.register("coi.js").then(() => {
    if (controlled) return;
    navigator.serviceWorker.addEventListener("controllerchange", () => location.reload(), { once: true });
  }).catch(() => {});
}
