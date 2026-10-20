import test, { type TestContext } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadPlugin } from "../../src/lib/plugins/loader.ts";
import type { PluginManifestWithDefaults } from "../../src/lib/plugins/manifest.ts";
import type { PluginContext } from "../../src/lib/plugins/index.ts";
const manifest: PluginManifestWithDefaults = {
  name: "lifecycle-test",
  version: "1.0.0",
  description: "Lifecycle regression",
  license: "MIT",
  main: "plugin.mjs",
  source: "local",
  tags: [],
  requires: { permissions: [] },
  hooks: {
    onRequest: true,
    onResponse: false,
    onError: false,
    onInstall: false,
    onActivate: false,
    onDeactivate: false,
    onUninstall: false,
    onStreamComplete: false,
  },
  skills: [],
  enabledByDefault: false,
  configSchema: {},
};
const ctx: PluginContext = {
  requestId: "probe",
  body: {},
  model: "test",
  provider: "test",
  metadata: {},
};
async function fixture(t: TestContext, source: string) {
  const dir = await mkdtemp(join(tmpdir(), "plugin-lifecycle-"));
  const file = join(dir, "plugin.mjs");
  await writeFile(file, source);
  const loaded = await loadPlugin(file, manifest);
  t.after(async () => {
    loaded.cleanup();
    await rm(dir, { recursive: true, force: true });
  });
  return loaded.plugin;
}
test("exit rejects future hook calls without waiting the hook timeout", async (t) => {
  const plugin = await fixture(t, "export function onRequest(){process.exit(0)}");
  await plugin.onRequest!(ctx);
  const start = performance.now();
  await plugin.onRequest!(ctx);
  assert.ok(performance.now() - start < 500, "dead process retained hook deadline");
});
test("disconnect rejects the pending call while child stays alive", async (t) => {
  const plugin = await fixture(
    t,
    "export function onRequest(){process.disconnect();setInterval(()=>{},1000);return new Promise(()=>{})}"
  );
  const start = performance.now();
  await plugin.onRequest!(ctx);
  assert.ok(performance.now() - start < 1000, "disconnected child retained hook deadline");
});
test("healthy child still serves concurrent hooks", async (t) => {
  const plugin = await fixture(t, "export async function onRequest(ctx){return {body:ctx.body}}");
  const results = await Promise.all([
    plugin.onRequest!({ ...ctx, body: { value: 1 } }),
    plugin.onRequest!({ ...ctx, body: { value: 2 } }),
  ]);
  assert.deepEqual(results, [{ body: { value: 1 } }, { body: { value: 2 } }]);
});
test("serialization failure clears its timer without poisoning a healthy worker", async (t) => {
  const plugin = await fixture(t, "export async function onRequest(ctx){return {body:ctx.body}}");
  await plugin.onRequest!(ctx);
  t.mock.timers.enable({ apis: ["setTimeout"] });
  await plugin.onRequest!({ ...ctx, body: { notJson: 1n } });
  t.mock.timers.tick(11000);
  assert.deepEqual(await plugin.onRequest!({ ...ctx, body: { ok: true } }), { body: { ok: true } });
  t.mock.timers.reset();
});
test("cleanup settles an in-flight hook immediately", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "plugin-cleanup-"));
  const entry = join(dir, "plugin.mjs");
  await writeFile(
    entry,
    "export async function onRequest(ctx){if(ctx.body.hang)return new Promise(()=>{});return {body:ctx.body}}"
  );
  const loaded = await loadPlugin(entry, manifest);
  t.after(async () => {
    loaded.cleanup();
    await rm(dir, { recursive: true, force: true });
  });
  await loaded.plugin.onRequest!(ctx);
  const pending = loaded.plugin.onRequest!({ ...ctx, body: { hang: true } });
  const start = performance.now();
  loaded.cleanup();
  await pending;
  assert.ok(performance.now() - start < 500);
});
